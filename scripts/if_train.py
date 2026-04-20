"""
IFEval GRPO training in native PyTorch.

Trains a model to follow instruction constraints (word count, formatting, keywords, etc.)
using Group Relative Policy Optimization with the Nemotron-Cascade-2 IF-RL dataset.

Usage:
  Single GPU:   uv run scripts/if_train.py
  Multi-GPU:    uv run torchrun --nproc_per_node=N scripts/if_train.py
  Dry run:      uv run scripts/if_train.py --dry-run
  Paper preset: uv run scripts/if_train.py --config configs/if_rl/nemotron_cascade2_paper.yaml
"""

import argparse
import json
import math
import os
import time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.common import DummyWandb, autodetect_device_type, compute_init, print0, compute_cleanup
from utils.openai_server import OpenAICompatibleRolloutClient, sync_server_model_weights
from tasks.IF_EVAL import summarize_constraint_evaluation

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
  
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "google/IFEval"
DATASET_CONFIG = "default"
MAX_PROMPT_LENGTH = 512

parser = argparse.ArgumentParser(description="IFEval GRPO training (native PyTorch)")
parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
parser.add_argument("--resume-from-checkpoint", type=str, default=None)
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")

# Generation
parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
parser.add_argument("--top-p", type=float, default=1.0, help="top-p sampling")
parser.add_argument("--max-new-tokens", type=int, default=512, help="max tokens to generate per sample")

# Training
parser.add_argument("--num-generations", type=int, default=16, help="number of generations per example/question")
parser.add_argument("--examples-per-step", type=int, default=8, help="total examples per optimization step across all ranks")
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over IFEval")

# Output
parser.add_argument("--report-dir", type=str, default=None, help="Directory for repo-level markdown reports")
parser.add_argument(
    "--sample-output-jsonl",
    nargs="?",
    const="__AUTO__",
    default=None,
    help="Optionally save training samples to JSONL. Defaults to <output_dir>/train_samples[.rankN].jsonl when enabled without a path.",
)

# Checkpointing
parser.add_argument("--output-dir", type=str, default="./ckpt_grpo_ifeval", help="output directory")
parser.add_argument("--save-every", type=int, default=50, help="save every N steps")
parser.add_argument("--eval-steps", type=int, default=50, help="eval every N steps")
parser.add_argument("--eval-size", type=int, default=50, help="eval size")
parser.add_argument("--seed", type=int, default=42, help="random seed")

# Optimization
parser.add_argument("--learning-rate", type=float, default=3e-6, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")

# vLLM server
parser.add_argument("--vllm-server-host", type=str, default="127.0.0.1", help="vLLM server host")
parser.add_argument("--vllm-server-port", type=int, default=8000, help="vLLM server port")
parser.add_argument("--vllm-model-name-for-requests", type=str, default="", help="vLLM model name for requests")
parser.add_argument("--vllm-api-key", type=str, default="EMPTY", help="vLLM API key")
parser.add_argument("--vllm-request-timeout", type=float, default=300.0, help="vLLM request timeout")
parser.add_argument("--vllm-sync-timeout", type=float, default=300.0, help="vLLM sync timeout")
parser.add_argument("--vllm-weight-sync-backend", type=str, default="nccl", help="vLLM weight sync backend")
parser.add_argument("--vllm-max-parallel-requests", type=int, default=8, help="vLLM max parallel requests")

args = parser.parse_args()
user_config = vars(args).copy()

def generate_completions(
    rollout_client: OpenAICompatibleRolloutClient,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_generations: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    return rollout_client.generate_completions(
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_generations=num_generations,
    )


def append_eval_log(output_dir: str, step: int, metrics: dict[str, float]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval_results.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# Evaluation Results\n\n")
            f.write("| Step | Timestamp | " + " | ".join(sorted(metrics.keys())) + " |\n")
            f.write("|------|-----------|" + "|".join("---" for _ in metrics) + "|\n")
        values = " | ".join(
            f"{metrics[k]:.4f}" if isinstance(metrics[k], float) else str(metrics[k])
            for k in sorted(metrics.keys())
        )
        f.write(f"| {step} | {timestamp} | {values} |\n")


def resolve_sample_output_jsonl_path(
    path_arg: str | None,
    output_dir: str,
    rank: int,
    world_size: int,
) -> str | None:
    if path_arg is None:
        return None

    if path_arg == "__AUTO__":
        filename = "train_samples.jsonl" if world_size == 1 else f"train_samples.rank{rank}.jsonl"
        return os.path.join(output_dir, filename)

    if world_size == 1:
        return path_arg

    root, ext = os.path.splitext(path_arg)
    if ext:
        return f"{root}.rank{rank}{ext}"
    return f"{path_arg}.rank{rank}"


def append_sample_rows(path: str | None, rows: list[dict]) -> None:
    if path is None or not rows:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def save_checkpoint(
    checkpoint_dir: str,
    raw_model,
    tokenizer,
    optimizer,
    scheduler,
    step: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    trainer_state = {
        "step": step,
        "seed": args.seed,
        "config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
            "examples_per_step": args.examples_per_step,
            "device_batch_size": args.device_batch_size,
        },
    }
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)


@torch.no_grad()
def iter_eval_records(
    rollout_client: OpenAICompatibleRolloutClient,
    tokenizer,
    eval_dataset: Dataset,
    ddp_world_size: int,
    ddp_rank: int,
):
    for idx in range(ddp_rank, len(eval_dataset), ddp_world_size):
        example = eval_dataset[idx]
        prompt = example["prompt"]
        instruction_ids = json.loads(example["instruction_id_list"])
        kwargs_list = json.loads(example["kwargs"])

        _, _, completions = generate_completions(
            rollout_client,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_generations=args.num_generations,
        )

        outcomes = []
        for completion in completions:
            summary = summarize_constraint_evaluation(completion, instruction_ids, kwargs_list)
            outcomes.append({
                "reward": summary["reward"],
                "all_passed": summary["all_passed"],
            })

        yield {
            "idx": idx,
            "outcomes": outcomes,
        }


@torch.no_grad()
def run_eval(
    raw_model,
    rollout_client: OpenAICompatibleRolloutClient,
    tokenizer,
    eval_dataset: Dataset | None,
    device: torch.device,
    ddp: bool,
    ddp_world_size: int,
    ddp_rank: int,
) -> dict[str, float] | None:
    if eval_dataset is None or len(eval_dataset) == 0:
        return None

    sync_server_model_weights(
        host=args.vllm_server_host,
        port=args.vllm_server_port,
        model=raw_model,
        backend=args.vllm_weight_sync_backend,
        timeout=args.vllm_sync_timeout,
        is_sync_leader=master_process,
        trainer_rank=ddp_rank,
        trainer_world_size=ddp_world_size,
    )
    if ddp:
        dist.barrier()

    reward_sum = torch.zeros(1, dtype=torch.float64, device=device)
    reward_sq_sum = torch.zeros(1, dtype=torch.float64, device=device)
    full_constraint_sum = torch.zeros(1, dtype=torch.float64, device=device)
    count = torch.zeros(1, dtype=torch.float64, device=device)

    for record in iter_eval_records(
        rollout_client,
        tokenizer,
        eval_dataset,
        ddp_world_size,
        ddp_rank,
    ):
        reward_tensor = torch.tensor(
            [outcome["reward"] for outcome in record["outcomes"]],
            dtype=torch.float64,
            device=device,
        )
        full_constraint_tensor = torch.tensor(
            [float(outcome["all_passed"]) for outcome in record["outcomes"]],
            dtype=torch.float64,
            device=device,
        )

        reward_sum += reward_tensor.sum()
        reward_sq_sum += (reward_tensor * reward_tensor).sum()
        full_constraint_sum += full_constraint_tensor.sum()
        count += reward_tensor.numel()

    if ddp:
        dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(full_constraint_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    mean_reward = (reward_sum / count.clamp(min=1)).item()
    if count.item() > 1:
        variance = ((reward_sq_sum - (reward_sum * reward_sum) / count) / (count - 1)).item()
    else:
        variance = 0.0
    std_reward = max(variance, 0.0) ** 0.5

    return {
        "eval/full_constraint_rate": (full_constraint_sum / count.clamp(min=1)).item(),
        "eval/mean_reward": mean_reward,
        "eval/reward_std": std_reward,
        "eval/num_completions": count.item(),
    }

device_type = autodetect_device_type()
if device_type != "cuda":
    raise ValueError("This native rewrite keeps vLLM rollouts and requires CUDA.")

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
torch.manual_seed(args.seed + ddp_rank)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="grpo-ifeval", name=args.run, config=user_config)

print0(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=HF_TOKEN)

print0(f"Loading training model: {MODEL_ID}...")
model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    dtype=model_dtype,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.to(device)

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

print0(
    f"Connecting to vLLM server at {args.vllm_server_host}:{args.vllm_server_port} "
    f"(model={MODEL_ID})..."
)
rollout_client = OpenAICompatibleRolloutClient(
    host=args.vllm_server_host,
    port=args.vllm_server_port,
    model_name=MODEL_ID,
    api_key=args.vllm_api_key,
    request_timeout=args.vllm_request_timeout,
    max_parallel_requests=args.vllm_max_parallel_requests,
)

def load_ifeval_dataset(eval_size: int) -> tuple[Dataset, Dataset | None]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

    rows = []
    for example in ds:
        prompt_text = (example.get("prompt") or "").strip()
        if not prompt_text:
            raise ValueError(f"No usable rows found in dataset {DATASET_NAME}/{DATASET_CONFIG}. Expected prompt field.")

        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=MAX_PROMPT_LENGTH)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        rows.append({
            "prompt": prompt_text,
            "instruction_id_list": json.dumps(example.get("instruction_id_list") or []),
            "kwargs": json.dumps(example.get("kwargs") or []),
        })

    full = Dataset.from_list(rows)
    if eval_size > 0 and eval_size < len(full):
        splits = full.train_test_split(test_size=eval_size, seed=42)
        return splits["train"], splits["test"]
    return full, None

train_dataset, eval_dataset = load_ifeval_dataset(args.eval_size)
print0(f"Train dataset: {len(train_dataset)} examples")
if eval_dataset:
    print0(f"Eval dataset: {len(eval_dataset)} examples (every {args.eval_steps} steps)")

import itertools

@torch.no_grad()
def get_batch():
    rank_indices = range(ddp_rank, len(train_dataset), ddp_world_size) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_dataset[example_idx]
        prompt = conversation["prompt"]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        instruction_ids = json.loads(conversation["instruction_id_list"])
        kwargs_list = json.loads(conversation["kwargs"])

        all_ids, all_masks, completions_text = generate_completions(
            rollout_client,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_generations=args.num_generations,
        )

        summaries = [
            summarize_constraint_evaluation(completion, instruction_ids, kwargs_list)
            for completion in completions_text
        ]
        rewards = torch.tensor(
            [summary["reward"] for summary in summaries],
            dtype=torch.float,
            device=device,
        )

        max_len = max(len(seq) for seq in all_ids)
        input_ids = []
        attention_masks = []
        completion_masks = []
        for seq, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)
            completion_masks.append(mask + [0] * pad_len)

        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long, device=device)
        completion_masks = torch.tensor(completion_masks, dtype=torch.long, device=device)

        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification
        targets[completion_masks[:, 1:] == 0] = -1 # -1 is the ignore index

        mu = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        advantages = (rewards - mu) / std
        
        yield (
            prompt,
            instruction_ids,
            kwargs_list,
            summaries,
            completions_text,
            inputs,
            targets,
            attention_masks,
            rewards,
            advantages,
        )

assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
    fused=(device.type == "cuda"),
)

num_steps = (len(train_dataset) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

def get_lr_lambda(step: int) -> float:
    return max(0.0, 1.0 - (step + 1) / max(num_steps, 1))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# data_iter = iter(loader)
sample_output_jsonl_path = resolve_sample_output_jsonl_path(
    args.sample_output_jsonl,
    args.output_dir,
    ddp_rank,
    ddp_world_size,
)


batch_iterator = get_batch()
run_start_time = time.perf_counter()
for step in range(num_steps):
    step_start_time = time.perf_counter()
       
    sync_server_model_weights(
        host=args.vllm_server_host,
        port=args.vllm_server_port,
        model=raw_model,
        backend=args.vllm_weight_sync_backend,
        timeout=args.vllm_sync_timeout,
        is_sync_leader=master_process,
        trainer_rank=ddp_rank,
        trainer_world_size=ddp_world_size,
    )
    if ddp:
        dist.barrier()

    if step > 0 and step % args.eval_steps == 0:
        model.eval()
        metrics = run_eval(
            raw_model,
            rollout_client,
            tokenizer,
            eval_dataset,
            device,
            ddp,
            ddp_world_size,
            ddp_rank,
        )
        if metrics:
            print0(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            wandb_run.log({"step": step, **metrics})
            append_eval_log(args.output_dir, step, metrics)

    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):   
        (
            prompt,
            instruction_ids,
            kwargs_list,
            summaries,
            completions_text,
            inputs_all,
            targets_all,
            attention_masks,
            rewards_all,
            advantages_all,
        ) = next(batch_iterator)
        
        total_seqs = inputs_all.shape[0]
        assert total_seqs % args.device_batch_size == 0
        num_passes = total_seqs // args.device_batch_size

        model.train()
        for pass_idx in range(num_passes):
            b0 = pass_idx * args.device_batch_size
            b1 = min(b0 + args.device_batch_size, total_seqs)
            
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            attention_mask = attention_masks[b0:b1, :-1]

            logits = model(input_ids=inputs, attention_mask=attention_mask).logits
            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
                ignore_index=-1,
            ).view_as(targets)
      
            pg_obj = (log_probs * advantages.unsqueeze(-1)).sum() 
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")

            sample_rows = []
            for local_idx, completion in enumerate(completions_text[b0:b1]):
                generation_idx = b0 + local_idx
                summary = summaries[generation_idx]
                sample_rows.append({
                    "step": step,
                    "example_step": example_step,
                    "pass_idx": pass_idx,
                    "pass_sample_index": local_idx,
                    "generation_index": generation_idx,
                    "prompt": prompt,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs_list,
                    "completion": completion,
                    "reward": rewards[local_idx].item(),
                    "advantage": advantages[local_idx].item(),
                    "all_passed": summary["all_passed"],
                    "constraint_results": summary["results"],
                })
            append_sample_rows(sample_output_jsonl_path, sample_rows)
            
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in completions_text)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)        
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    step_duration_sec = time.perf_counter() - step_start_time
    iter_per_sec = 1.0 / max(step_duration_sec, 1e-12)
    steps_remaining = max(num_steps - (step + 1), 0)
    eta_seconds = steps_remaining / max(iter_per_sec, 1e-12)

    print0(
        f"Step {step + 1}/{num_steps} | Average reward: {mean_reward} | "
        f"Average sequence length: {mean_sequence_length:.2f} | "
        f"iter/sec={iter_per_sec:.2f} | eta={format_duration(eta_seconds)}"
    )
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "iter_per_sec": iter_per_sec,
        "eta_seconds": eta_seconds,
        "runtime_seconds": time.perf_counter() - run_start_time,
        "lrm": get_lr_lambda(step),
        "lr": optimizer.param_groups[0]["lr"],
    })
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
        save_checkpoint(ckpt_dir, raw_model, tokenizer, optimizer, scheduler, step)
        print0(f"Saved checkpoint to {ckpt_dir}")

wandb_run.finish() 
compute_cleanup()
