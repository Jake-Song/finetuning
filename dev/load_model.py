import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}")
    print(f"Architecture: {model.config.architectures}")
    print(f"Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Dtype: {model.dtype}")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
