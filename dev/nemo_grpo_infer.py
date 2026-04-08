import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference utility for GRPO-trained HF checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory (e.g., checkpoint-100)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (defaults to --checkpoint)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for one-shot generation")
    parser.add_argument("--input-jsonl", type=str, default=None, help="Input JSONL for batch generation")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Output JSONL path for batch generation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Load dtype for model weights",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.prompt) == bool(args.input_jsonl):
        raise ValueError("Provide exactly one of --prompt or --input-jsonl")
    if args.input_jsonl and not args.output_jsonl:
        raise ValueError("--output-jsonl is required when --input-jsonl is set")
    if args.output_jsonl and not args.input_jsonl:
        raise ValueError("--output-jsonl can only be used with --input-jsonl")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_arg]


def load_model_and_tokenizer(
    checkpoint: str,
    tokenizer_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    tok_path = tokenizer_path or checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def normalize_prompt_input(tokenizer: Any, prompt_input: Any) -> str:
    if isinstance(prompt_input, str):
        return prompt_input

    if isinstance(prompt_input, list):
        if all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt_input):
            return tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
        if all(isinstance(entry, str) for entry in prompt_input):
            return "\n".join(prompt_input)

    raise ValueError(f"Unsupported prompt input format: {type(prompt_input)}")


def extract_prompt_from_item(item: dict[str, Any], tokenizer: Any) -> str:
    rcp = item.get("responses_create_params")
    if isinstance(rcp, dict) and "input" in rcp:
        return normalize_prompt_input(tokenizer, rcp["input"])

    for key in ("prompt", "input", "question", "text"):
        value = item.get(key)
        if isinstance(value, (str, list)):
            return normalize_prompt_input(tokenizer, value)

    raise ValueError("Could not extract a prompt from input row")


@torch.inference_mode()
def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_single_prompt(args: argparse.Namespace, model: Any, tokenizer: Any, device: torch.device) -> None:
    text = generate_one(
        model=model,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=device,
    )
    print(text)


def run_jsonl_batch(args: argparse.Namespace, model: Any, tokenizer: Any, device: torch.device) -> None:
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    total = 0
    failed = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            try:
                prompt_text = extract_prompt_from_item(row, tokenizer)
                generated_text = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    device=device,
                )
                result = {
                    **row,
                    "generated_text": generated_text,
                    "inference_meta": {
                        "checkpoint": args.checkpoint,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "do_sample": args.do_sample,
                    },
                }
            except Exception as exc:
                failed += 1
                result = {
                    **row,
                    "generated_text": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "inference_meta": {
                        "checkpoint": args.checkpoint,
                    },
                }
                print(f"WARNING line {line_idx}: {exc}")

            dst.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed = time.time() - started_at
    print(
        f"Finished JSONL inference: total={total}, failed={failed}, "
        f"elapsed={elapsed:.2f}s, output={output_path}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device={device}, dtype={dtype}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device, dtype)

    if args.prompt:
        run_single_prompt(args, model, tokenizer, device)
    else:
        run_jsonl_batch(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
