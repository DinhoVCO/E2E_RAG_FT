from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    QWEN3_LORA_TARGET_MODULES,
)


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN3_LORA_TARGET_MODULES,
    )


def load_qwen3_generative_with_lora(
    *,
    model_name: str = DEFAULT_BASE_MODEL,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()
    return model, tokenizer
