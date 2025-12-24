"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from prismatic.conf import ModelConfig, VLAConfig
from prismatic.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
)
from prismatic.models.vlms import VLA
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.tokenizer import TRAJECTORY_CONVERTER_REGISTRY

from prismatic.util.hf_utils import find_model_in_cache

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Model ID to complete snapshot path mapping
MODEL_ID_TO_HF_PATH = {
    "distilgpt2": "models--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "qwen3-vl-2b": "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
    "qwen3-vl-4b": "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "qwen3-vl-7b": "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/placeholder",  # TODO: Add real path when available
    "qwen3-vl-72b": "models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/placeholder",  # TODO: Add real path when available
}

# Path type constants
PATH_TYPE_CHECKPOINT = "checkpoints"
PATH_TYPE_HF_CACHED = "hf_hub_cached"


def _detect_model_path_type(model_id_or_path: Union[str, Path]) -> Tuple[str, Path]:
    """
    两种情况，训练的checkpoint和提前load的hf cache
    """
    model_id_or_path = str(model_id_or_path)
    # 1. Check if it's a known model ID (highest priority)
    if model_id_or_path in MODEL_ID_TO_HF_PATH:
        relative_path = MODEL_ID_TO_HF_PATH[model_id_or_path]
        resolved_path = Path(os.environ.get("HF_HOME")) / "hub" / relative_path
        if resolved_path.exists():
            return PATH_TYPE_HF_CACHED, resolved_path
    # 2. Check for checkpoint file (.safetensors)
    elif model_id_or_path.endswith(".safetensors"):
        return PATH_TYPE_CHECKPOINT, Path(model_id_or_path)
    # 3. 报错找不到模型
    else:
        model_ids = ", ".join(MODEL_ID_TO_HF_PATH.keys())
        raise ValueError(
            f"Model path `{model_id_or_path}` not found.\n\n"
            f"Supported Model IDs: {model_ids}\n\n"
            f"To download models, run:\n"
            f"  python download_models.py --model <model_id>"
        )


def create_trajectory_converter(vla_cfg: VLAConfig, tokenizer):
    """Create trajectory converter from VLA config."""
    converter_type = vla_cfg.trajectory_converter_type
    converter_n_bins = vla_cfg.trajectory_n_bins
    converter_n_dims = vla_cfg.trajectory_n_dims

    overwatch.info(
        f"Creating Trajectory Converter [bold]{converter_type}[/] "
        f"(n_bins={converter_n_bins}, n_dims={converter_n_dims})"
    )

    if converter_type not in TRAJECTORY_CONVERTER_REGISTRY:
        raise ValueError(
            f"Unknown trajectory converter type `{converter_type}`. "
            f"Available: {list(TRAJECTORY_CONVERTER_REGISTRY.keys())}"
        )

    return TRAJECTORY_CONVERTER_REGISTRY[converter_type](
        tokenizer=tokenizer,
        n_bins=converter_n_bins,
        n_dims=converter_n_dims,
    )


def _load_model_config(resolved_path: Path, vla_cfg: VLAConfig) -> dict:
    """
    Load model configuration from checkpoint directory or VLA config.

    Returns:
        Dictionary with model configuration fields
    """
    config_json = resolved_path / "config.json"

    # Try loading from config.json first
    if config_json.exists():
        with open(config_json, "r") as f:
            config_data = json.load(f)
        model_cfg = config_data.get("model", {})
        if model_cfg:
            return model_cfg

    # Fallback: reconstruct from base_vlm in VLA config
    base_vlm_id = vla_cfg.base_vlm
    if not base_vlm_id:
        raise ValueError(
            "Could not determine model configuration. "
            "No config.json found and vla_cfg.base_vlm is not set."
        )

    overwatch.info(f"Reconstructing model config from base_vlm: {base_vlm_id}")

    # Extract model ID from HF path if needed
    model_id = base_vlm_id.split("/")[-1] if "/" in base_vlm_id else base_vlm_id

    try:
        model_cfg_instance = ModelConfig.get_choice_class(model_id)()
        return {
            "model_id": getattr(model_cfg_instance, "model_id", model_id),
            "vision_backbone_id": getattr(
                model_cfg_instance, "vision_backbone_id", None
            ),
            "llm_backbone_id": getattr(model_cfg_instance, "llm_backbone_id", None),
            "arch_specifier": getattr(model_cfg_instance, "arch_specifier", "gelu-mlp"),
            "image_resize_strategy": getattr(
                model_cfg_instance, "image_resize_strategy", "letterbox"
            ),
            "llm_max_length": getattr(model_cfg_instance, "llm_max_length", 2048),
        }
    except Exception as e:
        raise ValueError(
            f"Could not load model configuration from ModelConfig registry. "
            f"base_vlm: {base_vlm_id}, error: {e}"
        )


# === Main Load Function ===
def load(
    vla_cfg: VLAConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> VLA:
    """
    Load pretrained OpenVLA model (offline mode).

    Supports loading from:
    - Checkpoint file: /path/to/checkpoints/model.pt
    - HF cached model: Model ID or HF path resolved to local cache

    Args:
        vla_cfg: VLA configuration with base_vlm and trajectory converter settings
        checkpoint_path: Optional checkpoint path (overrides vla_cfg.base_vlm)
        load_for_training: Whether to load for training (affects freezing)

    Returns:
        OpenVLA model instance with trajectory converter
    """
    # Determine which path to use
    model_path = checkpoint_path if checkpoint_path else vla_cfg.base_vlm

    # Detect and resolve path
    path_type, resolved_path = _detect_model_path_type(model_path)
    overwatch.info(f"Loading from {path_type}: {resolved_path}")

    # For HF cached models, resolved_path is already the snapshot directory
    if path_type == PATH_TYPE_HF_CACHED:
        model_dir = Path(resolved_path)
        checkpoint_safetensors = (
            model_dir / "checkpoints" / "latest-checkpoint.safetensors"
        )
        if not checkpoint_safetensors.exists():
            checkpoint_safetensors = None
    else:
        checkpoint_safetensors = checkpoint_path
        model_dir = (
            checkpoint_path.parent
            if "checkpoints" not in str(checkpoint_path)
            else checkpoint_path.parent.parent
        )

    # Load model configuration
    model_cfg = _load_model_config(model_dir, vla_cfg)

    overwatch.info(
        f"Model Configuration:\n"
        f"  Model ID:         [bold blue]{model_cfg.get('model_id')}[/]\n"
        f"  Vision Backbone:  [bold]{model_cfg.get('vision_backbone_id')}[/]\n"
        f"  LLM Backbone:     [bold]{model_cfg.get('llm_backbone_id')}[/]\n"
        f"  Arch Specifier:   [bold]{model_cfg.get('arch_specifier')}[/]\n"
        f"  Checkpoint:       [underline]{checkpoint_safetensors}[/]"
    )

    # Load base components (vision + LLM backbones)
    overwatch.info(
        f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]"
    )
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"], model_cfg["image_resize_strategy"]
    )

    overwatch.info(
        f"Loading LLM Backbone [bold]{model_cfg['llm_backbone_id']}[/] "
        f"(mode: {'training' if load_for_training else 'inference'})"
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        inference_mode=not load_for_training,
    )

    # Create trajectory converter
    trajectory_converter = create_trajectory_converter(vla_cfg, tokenizer)

    # Initialize OpenVLA model
    overwatch.info("Initializing OpenVLA model")
    if checkpoint_safetensors is not None:
        # Load from existing checkpoint
        vla = VLA.from_pretrained(
            checkpoint_safetensors,
            model_cfg["model_id"],
            vision_backbone,
            llm_backbone,
            trajectory_converter=trajectory_converter,
            arch_specifier=model_cfg["arch_specifier"],
            freeze_weights=not load_for_training,
        )
    else:
        # Create new OpenVLA model without checkpoint (fresh initialization)
        overwatch.info(
            "No checkpoint found; creating fresh OpenVLA model from backbones"
        )
        vla = VLA(
            model_cfg["model_id"],
            vision_backbone,
            llm_backbone,
            trajectory_converter=trajectory_converter,
            arch_specifier=model_cfg["arch_specifier"],
        )

    if not load_for_training:
        overwatch.info("Loading for inference, use evaluation mode.")
        vla.requires_grad_(False)
        vla.eval()

    overwatch.info("✅ Successfully loaded OpenVLA model")

    return vla
