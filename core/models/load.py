"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from omegaconf import OmegaConf

from core.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
)
from core.models.vlms import VLA, Qwen3VLA
from core.util.overwatch import initialize_overwatch
from core.vla.tokenizer import TRAJECTORY_CONVERTER_REGISTRY

from core.util.hf_utils import find_model_in_cache

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Model ID to complete snapshot path mapping
MODEL_ID_TO_HF_PATH = {
    "distilgpt2": "models--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "qwen2.5-0.5b": "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",  # Qwen2.5-0.5B (will be auto-downloaded if not cached)
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


def create_trajectory_converter(vla_cfg, tokenizer):
    """Create trajectory converter from VLA config."""
    # Handle both OmegaConf and regular config
    converter_type = vla_cfg.get('trajectory', {}).get('converter_type')
    converter_n_bins = vla_cfg.get('trajectory', {}).get('n_bins')
    converter_n_dims = vla_cfg.get('trajectory', {}).get('n_dims')

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



# === Main Load Function ===
def load(
    vla_cfg,  # Can be VLAConfig or OmegaConf DictConfig
    checkpoint_path: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> VLA:
    """
    Load pretrained OpenVLA model (offline mode).

    Supports loading from:
    - Checkpoint file: /path/to/checkpoints/model.pt
    - HF cached model: Model ID or HF path resolved to local cache

    Args:
        vla_cfg: VLA configuration with base_vlm and trajectory converter settings (VLAConfig or OmegaConf)
        checkpoint_path: Optional checkpoint path (overrides vla_cfg.base_vlm)
        load_for_training: Whether to load for training (affects freezing)

    Returns:
        OpenVLA model instance with trajectory converter
    """
    # Handle both OmegaConf DictConfig and VLAConfig
    base_vlm = vla_cfg.base_vlm if hasattr(vla_cfg, 'base_vlm') else vla_cfg.get('base_vlm')
    vla_id = vla_cfg.vla_id if hasattr(vla_cfg, 'vla_id') else vla_cfg.get('vla_id')
    
    overwatch.info(f"[DEBUG] vla_cfg type: {type(vla_cfg)}")
    overwatch.info(f"[DEBUG] vla_id: {vla_id}")
    overwatch.info(f"[DEBUG] base_vlm: {base_vlm}")

    # Determine which path to use
    model_path = checkpoint_path if checkpoint_path else base_vlm

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

    # Load model configuration from vla_cfg.vlm
    model_cfg = vla_cfg.get("vlm") if isinstance(vla_cfg, dict) else OmegaConf.to_container(vla_cfg.vlm)
    if model_cfg is None:
        model_cfg = {}
    
    # Ensure model_id is set from vla_cfg
    model_id = vla_cfg.get("model_id") if isinstance(vla_cfg, dict) else vla_cfg.model_id
    model_cfg["model_id"] = model_id

    overwatch.info(
        f"Model Configuration:\n"
        f"  Vision Backbone:  [bold]{model_cfg.get('vision_backbone_id')}[/]\n"
        f"  LLM Backbone:     [bold]{model_cfg.get('llm_backbone_id')}[/]\n"
        f"  Arch Specifier:   [bold]{model_cfg.get('arch_specifier')}[/]\n"
        f"  Checkpoint:       [underline]{checkpoint_safetensors}[/]"
    )

    # === Standard OpenVLA Branch: Separate vision + LLM + projector ===
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
        llm_max_length=model_cfg.get("llm_max_length"),
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
