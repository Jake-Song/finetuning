import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import requests
import yaml
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/opencodereasoning_filtered_25k_train.jsonl"
    eval_dataset_path: str = "data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl"

    # GRPO
    num_generations: int = 8
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_completion_length: int = 16384
    learning_rate: float = 1e-5
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    temperature: float = 1.0
    top_p: float = 0.999
    warmup_steps: int = 5
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0

    # infra
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    head_server_host: str = "127.0.0.1"
    head_server_port: int = 11000
    request_timeout: float = 10800.0

    max_steps: int = 1000
    output_dir: str = "./ckpt_grpo_code_gen"
    save_steps: int = 10
    logging_steps: int = 1
    seed: int = 42

    # wandb
    report_to: str = "none"
    task: str = "code_gen"


@dataclass
class NeMoGymGRPOConfig(GRPOConfig):
    agent_servers: dict[str, str] | None = None
    request_timeout: float = 10800


# -----------------------------
# NeMo Gym agent discovery
# -----------------------------
def get_agent_servers(host: str, port: int) -> dict[str, str]:
    response = requests.get(f"http://{host}:{port}/global_config_dict_yaml", timeout=10)
    response.raise_for_status()
    global_config = OmegaConf.create(yaml.safe_load(response.text))

    agent_servers = {}
    for server_name, server_config in global_config.items():
        if hasattr(server_config, "responses_api_agents"):
            agents = server_config.responses_api_agents
            for agent_key in agents.keys():
                agent_config = getattr(agents, agent_key)
                if hasattr(agent_config, "host") and hasattr(agent_config, "port"):
                    agent_host = agent_config.host
                    if agent_host in ("127.0.0.1", "0.0.0.0", "localhost"):
                        agent_host = host
                    agent_servers[agent_key] = f"http://{agent_host}:{agent_config.port}"

    if not agent_servers:
        raise ValueError("No agents found in global config")
    return agent_servers


# -----------------------------
# Reward
# -----------------------------
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    env_rewards = kwargs.get("env_reward")
    assert env_rewards is not None, "env_reward not found in kwargs"
    return [float(r) for r in env_rewards]


# -----------------------------
# Async agent rollout
# -----------------------------
async def call_nemo_gym_agents(
    prompts: list[str],
    dataset_items: list[dict[str, Any]],
    agent_servers: dict[str, str],
    timeout: float,
    max_completion_length: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.999,
) -> list[dict[str, Any]]:
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        tasks = []
        for prompt, item in zip(prompts, dataset_items, strict=False):
            request_body = item.copy()

            if "responses_create_params" not in request_body:
                request_body["responses_create_params"] = {
                    "input": [{"role": "user", "content": prompt}],
                }

            params = request_body["responses_create_params"]
            params.setdefault("max_output_tokens", max_completion_length)
            params["temperature"] = temperature
            params["top_p"] = top_p

            agent_ref = item.get("agent_ref", {})
            agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else None
            if not agent_name or agent_name not in agent_servers:
                raise ValueError(
                    f"Missing or invalid agent_ref. Got: {agent_ref}. Available: {list(agent_servers.keys())}"
                )
            agent_url = agent_servers[agent_name]

            task = session.post(
                f"{agent_url}/run",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, response in enumerate(responses):
            try:
                if isinstance(response, Exception):
                    raise response
                json_data = await response.json()
                if not isinstance(json_data, dict):
                    raise ValueError(f"Expected dict, got {type(json_data)}")
                results.append(json_data)
            except Exception as e:
                print(f"WARNING: Request {i} failed: {e}")
                results.append({"response": {"output": []}, "reward": 0.0, "error": str(e)})

        return results


# -----------------------------
# Rollout function for GRPOTrainer
# -----------------------------
def nemo_gym_rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    is_eval = not trainer.model.training
    num_generations = (
        trainer.args.num_generations_eval
        if is_eval and trainer.args.num_generations_eval
        else trainer.args.num_generations
    )
    dataset = trainer.eval_dataset if is_eval and trainer.eval_dataset is not None else trainer.train_dataset

    expanded_prompts = []
    expanded_dataset_items = []

    for idx_str in prompts:
        idx = int(idx_str)
        item = json.loads(dataset[idx]["metadata"])
        for _ in range(num_generations):
            expanded_prompts.append(idx_str)
            expanded_dataset_items.append(dict(item))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        responses = loop.run_until_complete(
            call_nemo_gym_agents(
                expanded_prompts,
                expanded_dataset_items,
                trainer.args.agent_servers,
                trainer.args.request_timeout,
                trainer.args.max_completion_length,
                temperature=trainer.args.temperature,
                top_p=trainer.args.top_p,
            )
        )
    finally:
        loop.close()

    tokenizer = trainer.processing_class

    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    env_mask: list[list[int]] = []
    logprobs: list[list[float]] = []
    env_rewards: list[float] = []
    num_turns_list: list[int] = []

    for i, response in enumerate(responses):
        eos_token_id = tokenizer.eos_token_id or 0

        if not isinstance(response, dict) or response.get("error"):
            rollout_failed = True
        else:
            output_items = response.get("response", {}).get("output", [])
            has_content = output_items and any(
                item.get("type") == "function_call"
                or (
                    item.get("type") == "message"
                    and any(
                        c.get("type") == "output_text" and c.get("text", "").strip()
                        for c in item.get("content", [])
                    )
                )
                for item in output_items
            )
            rollout_failed = not has_content

        if rollout_failed:
            prompt_ids.append([eos_token_id])
            completion_ids.append([eos_token_id])
            env_mask.append([0])
            logprobs.append([0.0])
            env_rewards.append(0.0)
            num_turns_list.append(0)
            continue

        episode_reward = response.get("reward", 0.0)
        output_items = response.get("response", {}).get("output", [])

        rollout_ids: list[int] = []
        rollout_mask: list[int] = []
        rollout_logprobs: list[float] = []

        seen_token_ids: list[int] = []
        first_prompt = None
        num_turns = 0

        for item in output_items:
            if "prompt_token_ids" not in item or "generation_token_ids" not in item:
                continue

            num_turns += 1
            item_prompt_ids = item["prompt_token_ids"]
            item_gen_ids = item["generation_token_ids"]
            item_logprobs = item.get("generation_log_probs", [])

            if first_prompt is None:
                first_prompt = item_prompt_ids
                seen_token_ids = list(item_prompt_ids)
            else:
                tool_result_tokens = []
                if len(item_prompt_ids) > len(seen_token_ids):
                    if item_prompt_ids[: len(seen_token_ids)] != seen_token_ids:
                        raise ValueError(
                            f"[Turn {num_turns}] Non-contiguous messages. "
                            f"Expected prefix len {len(seen_token_ids)}, got {len(item_prompt_ids)}"
                        )
                    tool_result_tokens = item_prompt_ids[len(seen_token_ids) :]

                if tool_result_tokens:
                    rollout_ids.extend(tool_result_tokens)
                    rollout_mask.extend([0] * len(tool_result_tokens))
                    rollout_logprobs.extend([0.0] * len(tool_result_tokens))

            rollout_ids.extend(item_gen_ids)
            rollout_mask.extend([1] * len(item_gen_ids))
            assert len(item_logprobs) == len(item_gen_ids), (
                f"Logprobs len {len(item_logprobs)} != gen len {len(item_gen_ids)}"
            )
            rollout_logprobs.extend(item_logprobs)
            seen_token_ids = list(item_prompt_ids) + list(item_gen_ids)

        if not rollout_ids or first_prompt is None:
            raise ValueError(f"Rollout {i} has no valid turns")

        prompt_ids.append(first_prompt)
        completion_ids.append(rollout_ids)
        env_mask.append(rollout_mask)
        logprobs.append(rollout_logprobs)
        env_rewards.append(episode_reward)
        num_turns_list.append(num_turns)

    if not prompt_ids:
        raise RuntimeError("No valid rollouts. Check NeMo Gym and vLLM logs.")

    if num_turns_list:
        trainer.log(
            {
                "num_turns_mean": sum(num_turns_list) / len(num_turns_list),
                "num_turns_min": min(num_turns_list),
                "num_turns_max": max(num_turns_list),
            }
        )

    unique_prompt_ids = prompt_ids[::num_generations]

    return {
        "prompt_ids": unique_prompt_ids,
        "completion_ids": completion_ids,
        "env_mask": env_mask,
        "logprobs": logprobs,
        "env_reward": env_rewards,
        "num_turns": num_turns_list,
    }


# -----------------------------
# Dataset
# -----------------------------
def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                data.append({"prompt": str(idx), "metadata": json.dumps(item)})
    return Dataset.from_list(data)


# -----------------------------
# Dry run
# -----------------------------
def dry_run(cfg: TrainConfig):
    print("=" * 60)
    print("GRPO DRY RUN")
    print("=" * 60)

    # config summary
    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  dataset:          {cfg.dataset_path}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.per_device_train_batch_size}")
    print(f"  grad_accum:       {cfg.gradient_accumulation_steps}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")

    # agent server discovery
    print(f"\n[Agent Servers]")
    try:
        servers = get_agent_servers(cfg.head_server_host, cfg.head_server_port)
        for name, url in servers.items():
            print(f"  {name}: {url}")
    except Exception as e:
        print(f"  (not reachable: {e})")
        print(f"  This is OK for dry-run. Start NeMo Gym servers before real training.")

    # dataset
    print(f"\n[Dataset]")
    if os.path.exists(cfg.dataset_path):
        dataset = load_dataset_from_jsonl(cfg.dataset_path)
        print(f"  samples: {len(dataset)}")
        first = json.loads(dataset[0]["metadata"])
        print(f"  first entry keys: {list(first.keys())}")
        if "responses_create_params" in first:
            rcp = first["responses_create_params"]
            if "input" in rcp and rcp["input"]:
                content = rcp["input"][0].get("content", "")
                print(f"  first prompt preview: {content[:120]}...")
        if "agent_ref" in first:
            print(f"  agent_ref: {first['agent_ref']}")
    else:
        print(f"  {cfg.dataset_path} not found.")
        print(f"  Prepare dataset with ng_prepare_data first.")

    # tokenizer
    print(f"\n[Tokenizer]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, truncation_side="left", padding_side="left"
        )
        test_messages = [{"role": "user", "content": "Write a function to sort a list."}]
        rendered = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(rendered)["input_ids"]
        print(f"  vocab_size:  {tokenizer.vocab_size}")
        print(f"  pad_token:   {tokenizer.pad_token}")
        print(f"  eos_token:   {tokenizer.eos_token}")
        print(f"  chat template test: {len(tokens)} tokens")
    except Exception as e:
        print(f"  Failed to load tokenizer: {e}")

    # GRPOConfig validation
    print(f"\n[GRPOConfig]")
    try:
        training_args = NeMoGymGRPOConfig(
            output_dir=cfg.output_dir,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=cfg.vllm_server_host,
            vllm_server_port=cfg.vllm_server_port,
            num_generations=cfg.num_generations,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_completion_length=cfg.max_completion_length,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            epsilon=cfg.epsilon,
            epsilon_high=cfg.epsilon_high,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            warmup_steps=cfg.warmup_steps,
            lr_scheduler_type=cfg.lr_scheduler_type,
            gradient_checkpointing=True,
            loss_type="grpo",
            mask_truncated_completions=True,
            vllm_importance_sampling_correction=True,
            optim="adamw_torch_fused",
            bf16=True,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            report_to=cfg.report_to,
            seed=cfg.seed,
        )
        print(f"  Config created successfully")
        print(f"  effective_batch = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * cfg.num_generations}")
    except Exception as e:
        print(f"  Config validation failed: {e}")

    print(f"\n{'=' * 60}")
    print("DRY RUN COMPLETE")
    print("=" * 60)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO training with NeMo Gym Code Gen")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--vllm-server-host", type=str, default=None)
    parser.add_argument("--head-server-host", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()

    # optional YAML override
    overrides = {}
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}

    if args.vllm_server_host:
        overrides["vllm_server_host"] = args.vllm_server_host
    if args.head_server_host:
        overrides["head_server_host"] = args.head_server_host

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, type(getattr(cfg, k))(v))

    if args.dry_run:
        dry_run(cfg)
        return

    # discover agent servers
    agent_servers = get_agent_servers(cfg.head_server_host, cfg.head_server_port)
    print(f"Discovered agent servers: {agent_servers}")

    # load dataset
    if cfg.dataset_path.endswith((".jsonl", ".json")):
        dataset = load_dataset_from_jsonl(cfg.dataset_path)
    else:
        dataset = load_dataset(cfg.dataset_path, split="train")
    print(f"Train dataset: {len(dataset)} examples")

    eval_dataset = None
    if cfg.eval_dataset_path and os.path.exists(cfg.eval_dataset_path):
        eval_dataset = load_dataset_from_jsonl(cfg.eval_dataset_path)
        print(f"Eval dataset: {len(eval_dataset)} examples")

    # training config
    training_args = NeMoGymGRPOConfig(
        output_dir=cfg.output_dir,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=cfg.vllm_server_host,
        vllm_server_port=cfg.vllm_server_port,
        num_generations=cfg.num_generations,
        num_generations_eval=1,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        gradient_checkpointing=True,
        loss_type="grpo",
        mask_truncated_completions=True,
        vllm_importance_sampling_correction=True,
        shuffle_dataset=False,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=cfg.report_to,
        seed=cfg.seed,
        model_init_kwargs={"torch_dtype": "auto"},
        agent_servers=agent_servers,
        request_timeout=cfg.request_timeout,
    )

    task_name = cfg.task or os.path.basename(cfg.dataset_path).replace(".jsonl", "")
    model_short = cfg.model_name.split("/")[-1]
    training_args.run_name = (
        f"{task_name}_{model_short}"
        f"_g{cfg.num_generations}"
        f"_bs{cfg.per_device_train_batch_size}"
        f"_ga{cfg.gradient_accumulation_steps}"
        f"_lr{cfg.learning_rate}"
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, truncation_side="left", padding_side="left"
    )

    # trainer
    trainer = GRPOTrainer(
        model=cfg.model_name,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        rollout_func=nemo_gym_rollout_func,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
