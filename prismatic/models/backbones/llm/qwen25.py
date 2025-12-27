"""
qwen25.py

Qwen2.5-0.5B Backbone adaptation for VLA.
"""

from typing import Sequence, Type

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder


# 注册模型信息
QWEN25_MODELS = {
    "qwen2.5-0.5b-instruct": {
        "llm_family": "qwen2",
        "llm_cls": Qwen2ForCausalLM,
        "hf_hub_path": "Qwen/Qwen2.5-0.5B-Instruct",
        # "hf_hub_path": "Qwen/Qwen2.5-7B",
    }
}


class Qwen25Backbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048, # Qwen 支持更长的上下文，建议设为 2048
        inference_mode: bool = False,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            inference_mode=inference_mode,
            # ✅ 核心修改：Qwen2 原生支持 Flash Attention 2，必须开启！
            # 这会带来巨大的速度提升和显存优化
            use_flash_attention_2=True, 
            **QWEN25_MODELS[llm_backbone_id],
        )

        # Qwen Tokenizer 处理：
        # Qwen2 的 tokenizer 通常有 pad token，但为了保险起见，如果为 None 则设为 EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 确保 pad_token_id 被正确设置到模型配置中
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        # [IMPORTANT] 同样不 Resize Embeddings，保持与 VLA 动作 Token 逻辑一致
        
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # Qwen 是指令模型，但在 VLA 训练中，如果你已经手动构造了 "USER: ... ASSISTANT: ..." 格式
        # 这里依然使用 PurePromptBuilder 保持原样。
        # 如果你想用 Qwen 官方的 Chat 模板 (<|im_start|>...)，这里需要写一个新的 Builder。
        # 暂时保持 PurePromptBuilder 以兼容你的 Datacollator。
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        # ✅ FSDP 核心修改：必须指向 Qwen 的 DecoderLayer
        # 只有指定对了这个，FSDP 才能正确地把每一层切开，节省显存
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        # ✅ 精度修改：4090 强力推荐使用 BF16 (BFloat16)
        # 比 FP16 更难溢出，训练 Loss 更稳定
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        # ✅ 路径修改：Qwen/Llama 架构的层在 model.layers 中，而不是 transformer.h
        return (self.llm.model.layers[-1], self.llm.lm_head)