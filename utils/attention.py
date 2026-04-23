from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM

from utils.common import print0


def load_causal_lm_with_attention(
    model_name_or_path: str,
    *,
    attn_implementation: str,
    fallback_attn_implementation: str = "sdpa",
    log_prefix: str = "training model",
    **from_pretrained_kwargs: Any,
):
    requested_attn = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=requested_attn,
            **from_pretrained_kwargs,
        )
    except (ImportError, ValueError) as exc:
        if requested_attn == fallback_attn_implementation:
            raise

        print0(
            f"Requested {log_prefix} attention backend '{requested_attn}' is unavailable; "
            f"falling back to '{fallback_attn_implementation}'. Reason: {exc}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=fallback_attn_implementation,
            **from_pretrained_kwargs,
        )
        resolved_attn = fallback_attn_implementation
    else:
        resolved_attn = requested_attn

    print0(
        f"Loaded {log_prefix} with attention backend '{resolved_attn}' "
        f"(requested '{requested_attn}')."
    )
    return model, resolved_attn
