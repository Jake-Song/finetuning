import argparse
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
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over IFEval")

# Output
parser.add_argument("--report-dir", type=str, default=None, help="Directory for repo-level markdown reports")

# Checkpointing
parser.add_argument("--output-dir", type=str, default="./ckpt_grpo_ifeval", help="output directory")
parser.add_argument("--save-every", type=int, default=50, help="save every N steps")
parser.add_argument("--eval-steps", type=int, default=50, help="eval every N steps")
parser.add_argument("--eval-size", type=int, default=50, help="eval size")
parser.add_argument("--seed", type=int, default=42, help="random seed")

# Optimization
parser.add_argument("--lr-scheduler-type", type=str, default="linear", help="learning rate scheduler type")
parser.add_argument("--learning-rate", type=float, default=3e-6, help="learning rate")
parser.add_argument("--warmup-steps", type=int, default=10, help="warmup steps")
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

print(user_config)