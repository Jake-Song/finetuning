import os
import torch
import torch.distributed as dist

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "meta-llama/Llama-3.1-8B"


def setup_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    mesh = init_device_mesh("cuda", (world_size,))
    return rank, local_rank, world_size, mesh


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_decoder_layers(model):
    # LlamaForCausalLM -> model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Could not find decoder layers at model.model.layers")


def main():
    rank, local_rank, world_size, mesh = setup_dist()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("Please set HF_TOKEN")

    if rank == 0:
        print(f"[1] Loading tokenizer/config for {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    config = AutoConfig.from_pretrained(MODEL_ID, token=token)

    if rank == 0:
        print("[2] Building empty model on meta device")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Sanity: should still be meta tensors
    meta_params = sum(int(p.device.type == "meta") for p in model.parameters())
    total_params = sum(1 for _ in model.parameters())

    if rank == 0:
        print(f"[ok] parameter tensors on meta: {meta_params}/{total_params}")
        print(f"[ok] total parameter count: {count_params(model):,}")

    # FSDP2 mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    if rank == 0:
        print("[3] Applying FSDP2 bottom-up")

    # shard each decoder block first
    layers = get_decoder_layers(model)
    for layer in layers:
        fully_shard(
            layer,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )

    # shard root module last
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )

    # verify tensors still meta after wrapping
    meta_params_after = sum(int(p.device.type == "meta") for p in model.parameters())

    if rank == 0:
        print(f"[ok] parameter tensors on meta after fully_shard: {meta_params_after}/{total_params}")

    if rank == 0:
        print("[4] Tokenizer smoke test")
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello in one sentence."},
        ]
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = " ".join(m["content"] for m in messages)
        enc = tokenizer(prompt, return_tensors="pt")
        print(f"[ok] prompt chars: {len(prompt)}")
        print(f"[ok] input_ids shape: {tuple(enc['input_ids'].shape)}")

        print("[5] Dry run complete")
        print("No weights were materialized. No forward was run.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()