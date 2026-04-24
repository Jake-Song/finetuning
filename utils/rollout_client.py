from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class OpenAICompatibleRolloutClient:
    host: str
    port: int
    model_name: str
    api_key: str = "EMPTY"
    request_timeout: float = 300.0
    max_parallel_requests: int = 8

    def __post_init__(self) -> None:
        self.server_base_url = f"http://{self.host}:{self.port}"
        self.client = OpenAI(
            base_url=f"{self.server_base_url}/v1",
            api_key=self.api_key,
            timeout=self.request_timeout,
        )

    def _generate_one(
        self,
        tokenizer,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            n=num_generations,
            logprobs=0,
            extra_body={"return_tokens_as_token_ids": True},
        )

        all_input_ids = []
        all_completion_masks = []
        all_texts = []
        for choice in response.choices:
            completion_ids = [int(tok.split(":", 1)[1]) for tok in choice.logprobs.tokens]
            seq_ids = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            text = choice.text
            reward_text = text.split("</think>", 1)[1].lstrip() if "</think>" in text else ""

            all_input_ids.append(seq_ids)
            all_completion_masks.append(mask)
            all_texts.append(reward_text)

        return all_input_ids, all_completion_masks, all_texts

    def generate_completions(
        self,
        tokenizer,
        prompt: str | list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            return [], [], []

        all_input_ids: list[list[int]] = []
        all_completion_masks: list[list[int]] = []
        all_texts: list[str] = []

        max_workers = min(self.max_parallel_requests, len(prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_one,
                    tokenizer,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_generations=num_generations,
                )
                for prompt_text in prompts
            ]

            for future in futures:
                input_ids, completion_masks, texts = future.result()
                all_input_ids.extend(input_ids)
                all_completion_masks.extend(completion_masks)
                all_texts.extend(texts)

        return all_input_ids, all_completion_masks, all_texts
