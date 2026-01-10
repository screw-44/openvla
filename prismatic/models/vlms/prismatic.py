"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.util.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from transformers import GenerationMixin

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    # Required for transformers 4.35+ generate() method compatibility
    # Indicates model doesn't use stateful caches (e.g., Mamba models do)
    _is_stateful = False

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(
                vision_backbone.embed_dim, llm_backbone.embed_dim
            )
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(
                vision_backbone.embed_dim, llm_backbone.embed_dim
            )
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(
                vision_backbone.embed_dim, llm_backbone.embed_dim
            )
        else:
            raise ValueError(
                f"PrismaticVLM with `{arch_specifier = }` is not supported!"
            )

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [
            chr(ord("A") + i) for i in range(26)
        ]:
            token_idx_list = self.llm_backbone.tokenizer.encode(
                trigger_string, add_special_tokens=False
            )
            assert (
                len(token_idx_list) == 1
            ), f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load checkpoint from safetensors format
        from safetensors import safe_open
        from pathlib import Path

        checkpoint_path = Path(pretrained_checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load from safetensors
        with safe_open(str(checkpoint_path), framework="pt", device=device) as f:
            model_state_dict = {k: f.get_tensor(k) for k in f.keys()}

        # Reconstruct nested_keys from flat keys 按照这个框架的标准
        nested_state = {}
        for key, tensor in model_state_dict.items():
            start_key = key.split('.')[0]
            nested_state.setdefault(start_key, {})[key[len(start_key)+1:]] = tensor

        assert (
            "projector" in nested_state
            and "llm_backbone" in nested_state
            and "vision_backbone" in nested_state
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector`, 'vision_backbone' AND `llm_backbone`!"
        overwatch.info("Successfully loaded checkpoint from safetensors")

        vlm.projector.load_state_dict(nested_state["projector"])
        vlm.llm_backbone.load_state_dict(nested_state["llm_backbone"])
        vlm.vision_backbone.load_state_dict(nested_state["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()
        overwatch.info("Loadding has finish on custom projector and llm_backbone")
        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(
                f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1
            )

        elif stage in {"finetune", "vla-train"}:
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(
                f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1
            )

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = [
                "vision_backbone",
                "projector",
                "llm_backbone",
            ]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1
            )

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = [
                "vision_backbone",
                "projector",
                "llm_backbone",
            ]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        else:
            raise ValueError(
                f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >"
            )


    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # Check if past_key_values contains valid cached data
        # Support both legacy tuple format and DynamicCache format
        # print(
        #     "==正在使用这个forward，是调用的了， " "其中pixel_values:",
        #     pixel_values,
        #     "\n input_ids:",
        #     input_ids,
        #     "\n past_key_values:",
        #     past_key_values
        # )
        # 第一次调用是有pixel-values，后续调用是没有的
        has_valid_past_key_values = past_key_values.get_seq_length() > 0 if past_key_values is not None else False
        
        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and has_valid_past_key_values:
            assert pixel_values==None, "这里pixel values一定是none"
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None, # check this
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            patch_features = self.vision_backbone(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (
                    projected_patch_embeddings.shape[0],
                    projected_patch_embeddings.shape[1],
                ),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[:, :1],
                    projected_patch_attention_mask,
                    attention_mask[:, 1:],
                ],
                dim=1,
            )

        # 这个删除调？
        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        # print("input ids:", input_ids[0])
        # if labels is not None:
        #     print("labels: is :", labels[0])
        # print("attention_map: ", attention_mask[0])
        if labels is not None:
            projected_patch_labels = torch.full(
                (
                    projected_patch_embeddings.shape[0],
                    projected_patch_embeddings.shape[1],
                ),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [
                    labels[:, :1],
                    projected_patch_labels,
                    labels[:, 1:],
                ],
                dim=1,
            )

        # === Add Unimodal Handling 一个batch中，有的样本是图像+文本。而另外的样本是纯文本，这时候需要进行区分 ===
        fused_embeddings = multimodal_embeddings
        fused_attention_mask = multimodal_attention_mask
        fused_labels = multimodal_labels

        # if overwatch.is_rank_zero():
        #     print("fused_attention_mask: ", fused_attention_mask[0], "shape:", fused_attention_mask[0].shape)
        #     print("fused input:", fused_embeddings[0], "shape:", fused_embeddings[0].shape)
        #     if labels is not None:
        #         print("fused_labels:", fused_labels[0], "shape", fused_labels[0].shape)

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return output

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 认为这个参数会被赋值
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""

        # Check if past_key_values contains valid cached data
        # Support both legacy tuple format and DynamicCache format
        seq_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        has_valid_past_key_values = seq_length > 0

        if has_valid_past_key_values:
            input_ids = input_ids[:, -1:]
            # Once we have valid past_key_values, we don't need pixel_values anymore
            # Vision tokens are already encoded in the KV cache
            pixel_values = None

            # NOTE: We do NOT update attention_mask here because HF's GenerationMixin
            # passes attention_mask with text-only length (e.g., 70), but our cache
            # contains multimodal tokens (text + vision patches, e.g., 325).
            # We need to reconstruct the correct attention_mask in forward() based on
            # the actual past_key_values length.

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    # 删除了，实际不会有这种情况
    def load_from_checkpoint(
        self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None
    ) -> None:
        raise NotImplementedError

    # 把generate的部分删除调了，训练直接用forward，推理在modeling vla里面。

    # @torch.inference_mode()
    # def generate_ids(
    #     self, image: Image, prompt_text: str, **kwargs: str
    # ) -> torch.LongTensor:
    #     # For now, only support generation with a batch size of 1 for simplicity
    #     image_transform, tokenizer = (
    #         self.vision_backbone.image_transform,
    #         self.llm_backbone.tokenizer,
    #     )

    #     # Prepare Inputs
    #     input_ids = tokenizer(
    #         prompt_text, truncation=True, return_tensors="pt"
    #     ).input_ids.to(self.device)
    #     pixel_values = image_transform(image)
    #     if isinstance(pixel_values, torch.Tensor):
    #         pixel_values = pixel_values[None, ...].to(self.device)
    #     elif isinstance(pixel_values, dict):
    #         pixel_values = {
    #             k: v[None, ...].to(self.device) for k, v in pixel_values.items()
    #         }
    #     else:
    #         raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    #     # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
    #     autocast_dtype = self.llm_backbone.half_precision_dtype
    #     with torch.autocast(
    #         "cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training
    #     ):
    #         generated_ids = super().generate(
    #             input_ids=input_ids,  # Shape: [1, seq]
    #             pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
    #             **kwargs,
    #         )
    #     return generated_ids

    # @torch.inference_mode()
    # def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
    #     generated_ids = self.generate_ids(image, prompt_text, **kwargs)

    #     tokenizer = self.llm_backbone.tokenizer
    #     input_ids = tokenizer(
    #         prompt_text, truncation=True, return_tensors="pt"
    #     ).input_ids.to(self.device)

    #     generated_text = tokenizer.decode(
    #         generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    #     ).strip()
    #     return generated_text

