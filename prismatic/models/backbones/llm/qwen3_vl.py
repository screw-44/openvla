"""
qwen3_vl.py

LLM backbone wrapper for Qwen3-VL unified multimodal models, used in Scheme A integration.
This treats Qwen-VL as the language backbone with native multimodal processing, bypassing separate
vision backbone usage. It provides the required interface expected by the training stack.
"""

from functools import partial
from typing import Callable, Optional, Sequence, Type

import torch
import torch.nn as nn
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen3VLForConditionalGeneration


from prismatic.models.backbones.llm.base_llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, VicunaV15ChatPromptBuilder




MODEL_ID_MAP = {
    "2B": "Qwen/Qwen3-VL-2B-Instruct",
    "7B": "Qwen/Qwen3-VL-7B-Instruct",
    "4B": "Qwen/Qwen3-VL-4B-Instruct",
}


class Qwen3VLBackbone(LLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        inference_mode: bool = False,
        model_size: str = "2B",
    ) -> None:
        super().__init__(llm_backbone_id)

        hf_model_id = MODEL_ID_MAP.get(model_size, MODEL_ID_MAP["2B"])

        # Load unified multimodal model (allows network access if not cached)
        # For training, we'll load in float16 then convert to float32 for compatibility
        # For inference, we use float16 for efficiency
        load_dtype = torch.float16
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=load_dtype,
            trust_remote_code=True,
        )

        # Convert to float32 for training to match training stack expectations
        if not inference_mode:
            self.llm = self.llm.to(torch.float32)

        # Processor includes tokenizer and image processor
        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.model_max_length = llm_max_length

        # Ensure PAD token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Disable cache during training to be compatible with gradient checkpointing
        self.llm.config.use_cache = False if not inference_mode else True

        # Cache common properties - use float32 for training, float16 for inference
        self._half_precision_dtype = torch.float32 if not inference_mode else torch.float16
        self.inference_mode = inference_mode
        # Try to infer a representative transformer layer class for FSDP auto-wrap
        try:
            self._transformer_layer_cls = self.llm.model.layers[0].__class__
        except Exception:
            self._transformer_layer_cls = nn.Module

    def get_tokenizer(self):
        return self.tokenizer

    def get_fsdp_wrapping_policy(self) -> Callable:
        try:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            return partial(transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls})
        except Exception:
            # Fallback: no auto-wrap
            return lambda module, recurse, nonwrapped_numel: False

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Standard input embedding from underlying model
        return self.llm.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Sequence[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Qwen-VL native multimodal args
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward through Qwen-VL (supports both text-only and multimodal)."""
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # Use Vicuna-like chat prompt builder for instruct-style prompts
        return VicunaV15ChatPromptBuilder

    @property
    def embed_dim(self) -> int:
        # Qwen3-VL stores hidden size in text_config
        if hasattr(self.llm.config, 'text_config') and hasattr(self.llm.config.text_config, 'hidden_size'):
            return self.llm.config.text_config.hidden_size
        # Fallback to standard hidden_size
        return self.llm.config.hidden_size

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return self._transformer_layer_cls

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self._half_precision_dtype

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        # Unfreeze last transformer block and LM head for last-layer finetuning
        modules = []
        try:
            modules.extend([self.llm.model.layers[-1], self.llm.lm_head])
        except Exception:
            pass
        return tuple(modules)
