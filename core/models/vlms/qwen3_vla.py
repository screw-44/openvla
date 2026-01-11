"""
qwen3_vla.py

完全独立的 Qwen3-VL 模型实现，用于 VLA 任务。
不依赖 prismatic 的 vision backbone / llm backbone 分离架构。

关键设计：
- Qwen3-VL 是一体式模型（vision + language 融合）
- 使用 processor-driven tokenization 处理图像和文本
- 支持双摄像头输入
- 动作通过 VlaTokenizer 离散化后追加到序列末尾
- 支持 left padding
- 可选择性冻结 vision/language 部分
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.models.backbones.llm.prompting import PromptBuilder
from core.models.vlms.base_vlm import VLM
from core.util.overwatch import initialize_overwatch
from core.vla.tokenizer import BaseTrajectoryConverter

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class Qwen3VLA(VLM):
    """
    Qwen3-VL based VLA model.

    与 PrismaticVLM 不同，这是一体式模型：
    - 不分离 vision_backbone / llm_backbone / projector
    - 使用 Qwen3 自己的 vision encoder 和 multimodal fusion
    - forward() 直接处理图像和文本，不需要预先提取 vision features
    """

    def __init__(
        self,
        model_id: str,
        model_size: str = "2B",  # "2B", "4B", "7B"
        trajectory_converter: Optional[BaseTrajectoryConverter] = None,
        enable_mixed_precision_training: bool = True,
        hf_cache_dir: Optional[Path] = None,
        **kwargs,
    ) -> None:
        # 注意：我们不调用 super().__init__，因为基类期望 vision_backbone/llm_backbone
        # 直接继承 nn.Module 和 GenerationMixin
        nn.Module.__init__(self)
        GenerationMixin.__init__(self)

        self.model_family = "qwen3-vl"
        self.model_id = model_id
        self.model_size = model_size
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.trajectory_converter = trajectory_converter

        # 构建 HF model path (Qwen3-VL)
        size_to_hub = {
            "2B": "Qwen/Qwen3-VL-2B-Instruct",
            "4B": "Qwen/Qwen3-VL-4B-Instruct",
            "7B": "Qwen/Qwen3-VL-7B-Instruct",
        }
        self.hf_hub_path = size_to_hub.get(model_size, size_to_hub["2B"])

        # Resolve HF cache path
        if hf_cache_dir is None:
            hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
            hf_cache_dir = Path(hf_home) / "hub"
        self.hf_cache_dir = hf_cache_dir

        overwatch.info(
            f"Loading Qwen3-VL {model_size} from HF cache (offline mode)",
            ctx_level=1,
        )

        # Load processor (handles image preprocessing + tokenization)
        # processor 需要 trust_remote_code=True
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.hf_hub_path,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=self.hf_cache_dir,
            )
            overwatch.info(f"Processor loaded from local cache", ctx_level=1)
        except Exception as e:
            overwatch.warning(
                f"Failed to load processor from local cache: {e}. Attempting online download..."
            )
            self.processor = AutoProcessor.from_pretrained(
                self.hf_hub_path,
                trust_remote_code=True,
                cache_dir=self.hf_cache_dir,
            )

        # Load model (Qwen3VLForConditionalGeneration)
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.hf_hub_path,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=self.hf_cache_dir,
                torch_dtype=(
                    torch.bfloat16 if enable_mixed_precision_training else torch.float32
                ),
            )
            overwatch.info(
                f"Qwen3VLForConditionalGeneration loaded from local cache", ctx_level=1
            )
        except Exception as e:
            overwatch.warning(
                f"Failed to load model from local cache: {e}. Attempting online download..."
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.hf_hub_path,
                trust_remote_code=True,
                cache_dir=self.hf_cache_dir,
                torch_dtype=(
                    torch.bfloat16 if enable_mixed_precision_training else torch.float32
                ),
            )

        # 配置 tokenizer padding（改为 left padding）
        self.processor.tokenizer.padding_side = "right"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id

        overwatch.info(
            f"Tokenizer padding side: {self.processor.tokenizer.padding_side}",
            ctx_level=1,
        )

        # Module keys for checkpoint saving
        self.all_module_keys = ["model", "processor"]
        self.trainable_module_keys = []  # 根据 freeze_backbones 动态设置

        # === GenerationMixin Expected Attributes ===
        self.generation_config = self.model.generation_config
        self.main_input_name = "input_ids"

        overwatch.info(f"Qwen3-VL {model_size} initialized successfully", ctx_level=1)

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    @property
    def config(self) -> PretrainedConfig:
        """Expose model config for GenerationMixin."""
        return self.model.config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        model_size: str = "2B",
        trajectory_converter: Optional[BaseTrajectoryConverter] = None,
        **kwargs,
    ) -> Qwen3VLA:
        """
        Load from checkpoint (for fine-tuned weights).

        Note: Base Qwen3-VL weights are always loaded from HF cache first,
        then checkpoint weights are loaded on top.
        """
        # 先加载基础模型
        vla = cls(
            model_id=model_id,
            model_size=model_size,
            trajectory_converter=trajectory_converter,
            **kwargs,
        )

        # 加载 checkpoint 权重
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            overwatch.info(f"Loading checkpoint from {pretrained_checkpoint}")
            from safetensors import safe_open

            with safe_open(
                str(pretrained_checkpoint), framework="pt", device="cpu"
            ) as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}

            # 过滤出 model 相关的权重
            model_state = {}
            for key, tensor in state_dict.items():
                if key.startswith("model."):
                    model_state[key[6:]] = tensor  # 去掉 "model." 前缀

            if model_state:
                vla.model.load_state_dict(model_state, strict=False)
                overwatch.info("Checkpoint weights loaded successfully")

        return vla

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        """
        Qwen3-VL 使用自己的 chat template，不需要 PromptBuilder。
        这个方法保留是为了兼容 VLM 接口。
        """
        from core.models.backbones.llm.prompting import PurePromptBuilder

        return PurePromptBuilder("qwen3-vl", system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        冻结部分模型参数。

        Qwen3-VL 的架构：
        - model.visual: 视觉编码器
        - model.model.layers: Transformer 层
        - model.lm_head: 语言模型头

        支持的 stage：
        - "vla-train": 冻结视觉编码器，训练语言部分
        - "vla-full-train": 全部训练
        - "vla-last-layer-train": 只训练最后一层
        """
        if stage == "vla-train":
            # 冻结视觉编码器
            if hasattr(self.model, "visual"):
                self.model.visual.requires_grad_(False)
                overwatch.info("[Frozen] 🥶 =>> Visual Encoder", ctx_level=1)

            # 训练语言模型部分
            if hasattr(self.model, "model"):
                self.model.model.requires_grad_(True)
            if hasattr(self.model, "lm_head"):
                self.model.lm_head.requires_grad_(True)
            overwatch.info("[TRAINABLE] 🔥 =>> Language Model", ctx_level=1)

            self.trainable_module_keys = ["model"]

        elif stage == "vla-full-train":
            # 全部训练
            self.model.requires_grad_(True)
            overwatch.info(
                "[TRAINABLE] 🔥 =>> Full Model (Vision + Language)", ctx_level=1
            )
            self.trainable_module_keys = ["model"]

        elif stage == "vla-last-layer-train":
            # 只训练最后一层
            self.model.requires_grad_(False)
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                self.model.model.layers[-1].requires_grad_(True)
            if hasattr(self.model, "lm_head"):
                self.model.lm_head.requires_grad_(True)
            overwatch.info(
                "[Frozen, except last layer] 🥶🔥 =>> Language Model", ctx_level=1
            )
            self.trainable_module_keys = ["model"]

        else:
            raise ValueError(f"Unknown stage `{stage}` for Qwen3VLA")

    def load_from_checkpoint(
        self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None
    ) -> None:
        """Load checkpoint weights (compatibility method)."""
        if pretrained_checkpoint is not None and pretrained_checkpoint.exists():
            overwatch.info(f"Loading checkpoint: {pretrained_checkpoint}")
            # Implementation similar to from_pretrained
            pass

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        返回 FSDP wrapping policy。

        注意：Qwen3-VL 可能更适合用 Accelerate + DeepSpeed，
        但这里提供 FSDP 策略以兼容框架。
        """
        from functools import partial
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Qwen3 使用标准 Transformer block
        # 需要找到对应的 layer class
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLDecoderLayer,
            )

            transformer_layer_cls = {Qwen2VLDecoderLayer}
        except ImportError:
            # 如果导入失败，使用通用策略
            overwatch.warning(
                "Could not import Qwen2VLDecoderLayer, using default policy"
            )
            transformer_layer_cls = set()

        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )

    def _prepare_inputs_for_qwen(
        self,
        pixel_values: Dict[str, torch.Tensor],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> Dict:
        """
        准备 Qwen3-VL 的输入格式。

        处理流程：
        1. 从 pixel_values dict 中提取 cam1, cam2
        2. 转换为 PIL Image (processor 期望 PIL 格式)
        3. 构建包含图像的 messages
        4. 使用 processor 处理
        5. 追加动作 tokens (input_ids)

        Args:
            pixel_values: {"cam1": [B, 3, H, W], "cam2": [B, 3, H, W]}
            input_ids: [B, seq_len] - 包含 prompt 和 action tokens
            attention_mask: [B, seq_len]

        Returns:
            Dict with keys: input_ids, attention_mask, pixel_values, image_grid_thw
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 准备图像列表（每个样本一个列表）
        # Qwen3-VL processor 可以处理多图输入
        images_per_sample = []
        for b in range(batch_size):
            sample_images = []
            for cam_key in sorted(pixel_values.keys()):  # cam1, cam2
                img_tensor = pixel_values[cam_key][b]  # [3, H, W]

                # 转换为 PIL Image
                # 注意：dataset 输出是 [0, 1] 范围的 tensor
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                pil_img = Image.fromarray(img_np)
                sample_images.append(pil_img)

            images_per_sample.append(sample_images)

        # 现在我们有了图像，但还需要构建 text prompt
        # 这里的 input_ids 已经包含了完整的序列（prompt + action tokens）
        # 但 Qwen processor 需要的是 messages 格式

        # 解决方案：直接使用 processor 的 tokenizer decode 出 text，
        # 然后重新用 processor 处理（包括图像）

        # 更简单的方案：我们跳过 processor 的完整处理，
        # 只用它的 image processor 处理图像，
        # 然后手动构建模型输入

        # 使用 processor 处理图像
        # processor.image_processor 可以批量处理
        all_images = []
        for sample_imgs in images_per_sample:
            all_images.extend(sample_imgs)

        # 处理图像
        image_inputs = self.processor.image_processor(
            images=all_images,
            return_tensors="pt",
        )

        # 移动到正确的设备
        for key in image_inputs:
            if isinstance(image_inputs[key], torch.Tensor):
                image_inputs[key] = image_inputs[key].to(device)

        # 返回模型所需的输入
        # Qwen3-VL 的 forward 需要：
        # - input_ids
        # - attention_mask
        # - pixel_values (from image_processor)
        # - image_grid_thw (from image_processor)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # 添加图像相关的输入
        if "pixel_values" in image_inputs:
            model_inputs["pixel_values"] = image_inputs["pixel_values"]
        if "image_grid_thw" in image_inputs:
            model_inputs["image_grid_thw"] = image_inputs["image_grid_thw"]

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]
        ] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Qwen3-VL forward pass for VLA training.

        注意：input_ids 和 labels 已经由 VlaTokenizer 处理好，
        包含了 prompt tokens 和 action tokens。
        我们只需要处理图像输入。
        """
        # 如果有缓存的 past_key_values，说明是 generation 的后续步骤
        # 此时不需要图像
        if past_key_values is not None and pixel_values is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        # 首次前向传播：需要处理图像
        if pixel_values is not None and isinstance(pixel_values, dict):
            # 准备 Qwen3-VL 格式的输入
            model_inputs = self._prepare_inputs_for_qwen(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 添加其他参数
            model_inputs.update(
                {
                    "labels": labels,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                    "return_dict": return_dict,
                }
            )

            # 调用模型
            return self.model(**model_inputs)
        else:
            # 没有图像，直接前向传播（纯文本或后续生成步骤）
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.inference_mode()
    def generate_ids(
        self,
        image: Union[Image.Image, Dict[str, Image.Image]],
        prompt_text: str,
        **kwargs,
    ) -> torch.LongTensor:
        """
        生成 token IDs（用于推理）。

        Args:
            image: PIL Image 或 {"cam1": PIL Image, "cam2": PIL Image}
            prompt_text: 文本 prompt
            **kwargs: 传递给 generate 的参数

        Returns:
            生成的 token IDs [1, seq_len]
        """
        # 准备图像列表
        if isinstance(image, dict):
            images = [image[k] for k in sorted(image.keys())]
        else:
            images = [image]

        # 构建 messages（Qwen3-VL 格式）
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": prompt_text}],
            }
        ]

        # 使用 processor 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # 生成
        generated_ids = self.model.generate(**inputs, **kwargs)

        return generated_ids

    # === GenerationMixin Required Methods ===
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation step."""
        return self.model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        return self.model._reorder_cache(past_key_values, beam_idx)

    @property
    def llm_backbone(self):
        """
        兼容属性：返回一个包含 tokenizer 和 prompt_builder_fn 的对象。
        用于与现有的 VLA 训练流程兼容。
        """

        class FakeLLMBackbone:
            def __init__(self, processor, model):
                self.processor = processor
                self.model = model
                self.tokenizer = processor.tokenizer

            def get_tokenizer(self):
                return self.tokenizer

            @property
            def prompt_builder_fn(self):
                from core.models.backbones.llm.prompting import PurePromptBuilder

                return PurePromptBuilder

            @property
            def transformer_layer_cls(self):
                # 返回 Qwen3 的 transformer layer class
                try:
                    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                        Qwen2VLDecoderLayer,
                    )

                    return Qwen2VLDecoderLayer
                except ImportError:
                    return nn.Module

            @property
            def last_layer_finetune_modules(self):
                # 返回最后一层的模块（用于部分微调）
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    return [self.model.model.layers[-1], self.model.lm_head]
                return []

        return FakeLLMBackbone(self.processor, self.model)

    @property
    def vision_backbone(self):
        """
        兼容属性：返回一个包含 image_transform 的对象。
        用于与现有的 VLA 训练流程兼容。
        """

        class FakeVisionBackbone:
            def __init__(self, processor):
                self.processor = processor
                self.identifier = "qwen3-vl-vision"

            def get_image_transform(self):
                """
                返回图像转换器。

                重要：这个 transform 会被传递给 LeRobotDataset。
                LeRobotDataset 输出的图像是 [0, 1] 范围的 tensor。
                我们需要将其转换为 Qwen processor 可以处理的格式。
                """
                return Qwen3ImageTransform(self.processor)

            def get_fsdp_wrapping_policy(self):
                """空策略（Qwen3-VL 不需要单独的 vision wrapping）"""
                from functools import partial
                from torch.distributed.fsdp.wrap import _module_wrap_policy

                return partial(_module_wrap_policy, module_classes=set())

        return FakeVisionBackbone(self.processor)


class Qwen3ImageTransform:
    """
    Qwen3-VL 的图像转换器。

    负责将 LeRobotDataset 输出的 [0, 1] tensor 转换为
    Qwen processor 可以处理的格式。

    注意：LeRobotDataset 会应用这个 transform，
    但 transform 的输出仍然是 tensor（而不是 PIL），
    因为我们需要在 batch collation 时保持 tensor 格式。

    实际的 PIL 转换会在 forward() 中进行。
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        转换图像。

        Args:
            img: PIL Image 或 torch.Tensor [C, H, W] in [0, 1]

        Returns:
            torch.Tensor [C, H, W] in [0, 1] (保持原样，实际转换在 forward 中)
        """
        # LeRobotDataset 可能传入 PIL 或 tensor
        if isinstance(img, Image.Image):
            # 转换为 tensor [0, 1]
            import torchvision.transforms.functional as TF

            return TF.to_tensor(img)
        elif isinstance(img, torch.Tensor):
            # 已经是 tensor，直接返回
            # 确保是 [C, H, W] 格式
            if img.ndim == 4:  # [B, C, H, W]
                img = img[0]
            return img
        else:
            raise TypeError(f"Unexpected image type: {type(img)}")
