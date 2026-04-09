from __future__ import annotations

import torch


def sync_vllm_model_weights(llm, model) -> None:
    """
    Sync a HF/torch model into a vLLM `LLM` instance using the public
    `apply_model` hook instead of private engine internals.

    This is compatible with newer vLLM releases where internal attributes like
    `llm_engine.model_executor.driver_worker` are no longer exposed.
    """

    weights = [(name, param.detach().cpu()) for name, param in model.named_parameters()]

    def _load_weights(vllm_model, named_weights):
        loaded = vllm_model.load_weights(named_weights)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return 0 if loaded is None else len(loaded)

    llm.apply_model(lambda vllm_model: _load_weights(vllm_model, weights))
