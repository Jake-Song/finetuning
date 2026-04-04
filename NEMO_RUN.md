# Run workflow

## 1. Prepare dataset

From the Gym repo:

```bash
cd Gym/
ng_prepare_data \
  "+config_paths=[resources_servers/code_gen/configs/code_gen.yaml]" \
  +output_dirpath=data/code_gen \
  +mode=train_preparation \
  +should_download=true \
  +data_source=huggingface
```

## 2. Start NeMo Gym servers (Terminal 1)

```bash
cd Gym/
ng_run "+config_paths=[resources_servers/code_gen/configs/code_gen.yaml,responses_api_models/vllm_model/configs/vllm_model_for_training.yaml]"
```

## 3. Start vLLM (Terminal 2)

```bash
uv run trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --host 0.0.0.0 \
  --port 8000
```

## 4. Train (Terminal 3)

From this repo (`finetuning`):

```bash
uv run python grpo_train.py
```
