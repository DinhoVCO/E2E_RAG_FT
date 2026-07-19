from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tesis_unicamp.generation.base import BaseGenerator, GenerationConfig

DEFAULT_MAX_LORA_RANK = 16


def configure_vllm_multiprocessing() -> None:
    """Use spawn for vLLM workers to avoid CUDA re-init errors after fork."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest


class VLLMOfflineGenerator(BaseGenerator):
    """Generate text in-process with vLLM."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        llm: LLM | None = None,
        lora_path: str | None = None,
        lora_name: str = "adapter",
        lora_int_id: int = 1,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
        system_prompt: str | None = None,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        **llm_kwargs: object,
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._lora_path = lora_path
        self._lora_name = lora_name
        self._lora_int_id = lora_int_id
        self._max_lora_rank = max_lora_rank
        self._system_prompt = system_prompt
        self._use_chat_template = use_chat_template
        self._enable_thinking = enable_thinking
        self._llm_kwargs = llm_kwargs
        self._lora_request: LoRARequest | None = None

    def _get_llm(self) -> LLM:
        if self._llm is None:
            from vllm import LLM

            llm_kwargs = dict(self._llm_kwargs)
            if self._lora_path is not None:
                llm_kwargs.setdefault("enable_lora", True)
                llm_kwargs.setdefault("max_lora_rank", self._max_lora_rank)
            self._llm = LLM(model=self.model_name, **llm_kwargs)
        return self._llm

    def _get_lora_request(self) -> LoRARequest | None:
        if self._lora_path is None:
            return None
        if self._lora_request is None:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(
                self._lora_name,
                self._lora_int_id,
                self._lora_path,
            )
        return self._lora_request

    def warmup(self) -> None:
        """Load the vLLM engine before any fork-based multiprocessing runs."""
        self._get_llm()

    def get_tokenizer(self):
        return self._get_llm().get_tokenizer()

    def count_tokens(self, text: str) -> int:
        tokenizer = self.get_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_formatted_prompt_tokens(self, user_content: str) -> int:
        formatted = self._format_prompts([user_content])[0]
        return self.count_tokens(formatted)

    def get_max_model_len(self) -> int:
        llm = self._get_llm()
        model_config = getattr(llm.llm_engine, "model_config", None)
        if model_config is not None and hasattr(model_config, "max_model_len"):
            return int(model_config.max_model_len)
        return int(self._llm_kwargs.get("max_model_len", 40960))

    def get_default_max_prompt_tokens(self, *, safety_margin: int = 256) -> int:
        return max(
            1024,
            self.get_max_model_len() - self.config.max_tokens - safety_margin,
        )

    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        tokenizer = self.get_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def _format_prompts(self, user_contents: list[str]) -> list[str]:
        if not self._use_chat_template:
            return user_contents

        llm = self._get_llm()
        tokenizer = llm.get_tokenizer()
        prompts: list[str] = []
        for content in user_contents:
            if self._system_prompt is None:
                messages = [{"role": "user", "content": content}]
            else:
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": content},
                ]
            prompt = self._apply_chat_template(tokenizer, messages)
            prompts.append(prompt)
        return prompts

    def _apply_chat_template(self, tokenizer, messages: list[dict[str, str]]) -> str:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self._enable_thinking,
        }
        try:
            return tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **template_kwargs)

    def _sampling_params(self, *, n: int | None = None) -> SamplingParams:
        from vllm import SamplingParams

        num_completions = n if n is not None else self.config.n
        kwargs: dict[str, object] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": num_completions,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        if self.config.stop:
            kwargs["stop"] = list(self.config.stop)
        return SamplingParams(**kwargs)

    def _generate_raw(self, formatted_prompts: list[str], *, n: int | None = None):
        sampling_params = self._sampling_params(n=n)
        lora_request = self._get_lora_request()
        if lora_request is None:
            return self._get_llm().generate(formatted_prompts, sampling_params)
        return self._get_llm().generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_request,
        )

    def generate_texts_multi(
        self,
        prompts: list[str],
        *,
        n: int | None = None,
    ) -> list[list[str]]:
        """Generate ``n`` completions per prompt (HyDE-style multi-sampling)."""
        if not prompts:
            return []

        num_completions = n if n is not None else self.config.n
        outputs = self._generate_raw(self._format_prompts(prompts), n=num_completions)
        return [
            [completion.text.strip() for completion in output.outputs]
            for output in outputs
        ]

    def generate_texts(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        outputs = self._generate_raw(self._format_prompts(prompts))
        return [output.outputs[0].text.strip() for output in outputs]
