"""
distilgpt2.py

Lightweight Causal LLM backbone using DistilGPT2 for fast debugging. This is a few-hundred MB model
that fits easily on a single GPU and provides the standard HF CausalLM API.
"""

from typing import Sequence, Type

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder


DISTILGPT2_MODELS = {
    "distilgpt2": {
        "llm_family": "gpt2",
        "llm_cls": GPT2LMHeadModel,
        "hf_hub_path": "distilgpt2",
    }
}


class DistilGPT2Backbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 1024,
        inference_mode: bool = False,
    ) -> None:
        # HACK: BREAKING 一定一定要是true，否则无法收敛（极端重要，千万不能修改
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            inference_mode=inference_mode,
            use_flash_attention_2=False, # 不能开启，gpt2会变成bidirectional的不知道为什么，很沙比（但是训练又受到影响了）
            **DISTILGPT2_MODELS[llm_backbone_id],
        )

        # GPT2 tokenizers have no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        # [IMPORTANT] For VLA training, we do NOT resize embeddings!
        # Action tokens use the LAST n_bins tokens of the existing vocabulary.
        # The ActionTokenizer assumes: action_token_begin_idx = vocab_size - (n_bins + 1)
        # So we must keep the model's vocab_size equal to the tokenizer's vocab_size.

    # NOTE: 这代表使用了哪一种prompting的具体实现
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # Use a simple non-chat prompt builder
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GPT2Block

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16 # 因为开不了flash attn，采用fp32训练

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.transformer.h[-1], self.llm.lm_head)
