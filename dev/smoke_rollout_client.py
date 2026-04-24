"""Smoke test for utils.rollout_client.OpenAICompatibleRolloutClient.

Prereq: vLLM server running, e.g. `bash setup/start_vllm.sh`.
Run: uv --project envs/trainer run python dev/smoke_rollout_client.py
"""
from __future__ import annotations

import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.rollout_client import OpenAICompatibleRolloutClient


MODEL_NAME = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
HOST = os.environ.get("SMOKE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SMOKE_PORT", "8000"))
PROMPT = "What is 17 * 23? Give the final numeric answer."
NUM_GENS = 2
MAX_NEW_TOKENS = 512


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    client = OpenAICompatibleRolloutClient(host=HOST, port=PORT, model_name=MODEL_NAME)

    input_ids, masks, texts = client.generate_completions(
        tokenizer,
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=0.95,
        num_generations=NUM_GENS,
    )

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    assert len(input_ids) == NUM_GENS, f"expected {NUM_GENS} rows, got {len(input_ids)}"
    assert len(masks) == NUM_GENS and len(texts) == NUM_GENS

    for i, (ids, mask, reward_text) in enumerate(zip(input_ids, masks, texts)):
        assert len(ids) == len(mask), f"row {i}: ids/mask length mismatch"
        completion_len = len(ids) - prompt_len
        assert completion_len > 0, f"row {i}: empty completion"
        assert sum(mask) == completion_len, f"row {i}: mask sum {sum(mask)} != {completion_len}"
        assert all(m == 0 for m in mask[:prompt_len]), f"row {i}: prompt positions not zeroed"
        assert all(m == 1 for m in mask[prompt_len:]), f"row {i}: completion positions not one"

        decoded = tokenizer.decode(ids[prompt_len:])
        assert "</think>" in decoded, f"row {i}: no </think> in completion decode:\n{decoded[:500]}"

        assert "<think>" not in reward_text and "</think>" not in reward_text, (
            f"row {i}: reward text still contains think tags:\n{reward_text[:500]}"
        )

        print(f"--- row {i} ---")
        print(f"  completion tokens : {completion_len}")
        print(f"  mask coverage     : {sum(mask)}/{len(mask)} (prompt_len={prompt_len})")
        print(f"  reward text (head): {reward_text[:200]!r}")
        print(f"  decoded (tail)    : {decoded[-200:]!r}")

    print("\nOK: thinking tokens preserved in input_ids, stripped from reward text.")


if __name__ == "__main__":
    main()
