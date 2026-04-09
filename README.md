# Finetuning Runbook

This repo trains RL policies in native PyTorch and uses a separate vLLM 0.19 OpenAI-compatible server for rollout generation.

## 1. Install dependencies

```bash
uv sync
```

If you need to add the OpenAI client separately:

```bash
uv add openai
```

## 2. Start the vLLM server

Use the helper script:

```bash
./setup/start_vllm.sh Qwen/Qwen2.5-1.5B-Instruct 8192 0.0.0.0 8000 nccl
```

Arguments:

1. model name or path
2. max model length
3. host
4. port
5. weight-sync backend (`nccl` or `ipc`)

The training scripts default to `127.0.0.1:8000`, so keep the server host and port aligned with the trainer flags.

## 3. Run a dry run first

Dry runs validate config, dataset loading, tokenizer loading, and reward logic without training.

```bash
uv run python scripts/if_train.py --dry-run
uv run python scripts/swe_rl_train.py --dry-run
uv run python scripts/multi_domain_rl_train.py --dry-run
```

## 4. Run training

### IF-RL

```bash
uv run python scripts/if_train.py \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000
```

### SWE-RL

```bash
uv run python scripts/swe_rl_train.py \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000
```

### Multi-domain RL

```bash
uv run python scripts/multi_domain_rl_train.py \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000
```

## 5. Multi-GPU training

Launch the trainer with `torchrun` and keep the same server settings:

```bash
uv run torchrun --nproc_per_node=2 scripts/if_train.py \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000
```

Use the same pattern for the SWE and multi-domain trainers.

## 6. Resume from a checkpoint

Each trainer saves HF checkpoints plus native optimizer/scheduler state under `output_dir/step_<n>` and `output_dir/final`.

```bash
uv run python scripts/if_train.py \
  --resume-from-checkpoint ./ckpt_grpo_ifeval/step_50 \
  --vllm-server-host 127.0.0.1 \
  --vllm-server-port 8000
```

## 7. Common server flags

The trainers support these server-related options:

```bash
--vllm-server-host
--vllm-server-port
--vllm-model-name
--vllm-api-key
--vllm-request-timeout
--vllm-sync-timeout
--vllm-weight-sync-backend
```

Use `--vllm-model-name` if the server model name differs from the trainer model name.

## 8. Notes

- The server is used for rollout generation through the OpenAI-compatible Chat Completions API.
- Weight sync is done separately through vLLM’s training/weight-transfer path.
- If training stalls, check that the server is reachable and that the weight-sync backend matches the server startup mode.
