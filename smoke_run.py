import os
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    mesh = init_device_mesh("cuda", (world_size,))

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config).cuda()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
    labels = torch.randint(0, config.vocab_size, (2, 128), device="cuda")

    for step in range(10):
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optim.step()
        if rank == 0:
            print(f"step {step+1:02d}/10 loss: {loss.detach().item():.4f}")

    if rank == 0:
        print("tiny FSDP2 smoke test passed")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()