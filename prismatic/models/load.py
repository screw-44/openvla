"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import torch

from prismatic.conf import ModelConfig, VLAConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.tokenizer import TRAJECTORY_CONVERTER_REGISTRY

from prismatic.util.hf_utils import find_model_in_cache

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"

# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())

def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")
    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))
    return description

# === Helper Functions for Path Detection ===
def _resolve_hf_hub_path_to_local_cache(hf_hub_path: str) -> Optional[Path]:
    """
    Attempt to resolve a HuggingFace Hub path (e.g., 'TRI-ML/prismatic-vlms/siglip-224px+7b') 
    to a local cache directory.
    
    This function checks HF_HOME environment variable and looks for the cached model.
    Returns None if not found locally.
    """
    # Extract repo_id and subfolder from HF Hub path
    # Format: "owner/repo_id/subfolder" or "owner/repo_id"
    parts = hf_hub_path.split("/")
    if len(parts) < 2:
        return None
    
    # Reconstruct repo_id (e.g., "TRI-ML/prismatic-vlms")
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
    
    # Normalize repo_id for HF cache directory format (replace "/" with "--")
    # e.g., "TRI-ML/prismatic-vlms" -> "models--TRI-ML--prismatic-vlms"
    cache_dir_name = f"models--{repo_id.replace('/', '--')}"
    
    # Check HF_HOME or use default
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    candidate_path = Path(hf_home) / "hub" / cache_dir_name
    
    if not candidate_path.exists():
        return None
    
    # Look for the latest snapshot
    snapshots_dir = candidate_path / "snapshots"
    if not snapshots_dir.exists() or not list(snapshots_dir.iterdir()):
        return None
    
    # Get the first (and typically only) snapshot directory
    snapshot_dir = next(iter(snapshots_dir.iterdir()))
    
    # If there's a subfolder, append it
    if subfolder:
        full_path = snapshot_dir / subfolder
    else:
        full_path = snapshot_dir
    
    return full_path if full_path.exists() else None


def _detect_model_path_type(model_id_or_path: Union[str, Path]) -> tuple[str, Union[str, Path]]:
    """
    Detect the type of model path and return (path_type, resolved_path).
    
    Supported path types:
    - "checkpoint": Local checkpoint file (.pt)
    - "local_dir": Local directory with config.json and checkpoints/
    - "hf_hub_cached": HuggingFace Hub path that resolves to local cache
    
    Note: Supports HF Hub paths like 'TRI-ML/prismatic-vlms/siglip-224px+7b' but requires
    the model to be pre-cached locally (e.g., in HF_HOME).
    """
    # Convert to Path if string and check if it's a file path
    if isinstance(model_id_or_path, str):
        if model_id_or_path.endswith(".pt") or model_id_or_path.startswith("/"):
            path_obj = Path(model_id_or_path)
            if path_obj.is_file() and path_obj.suffix == ".pt":
                return "checkpoint", path_obj
    elif isinstance(model_id_or_path, Path):
        if model_id_or_path.is_file() and model_id_or_path.suffix == ".pt":
            return "checkpoint", model_id_or_path
    
    # Check if it's a local directory
    path_obj = Path(model_id_or_path)
    if path_obj.is_dir():
        return "local_dir", path_obj
    
    # Try to resolve as HF Hub path to local cache
    if isinstance(model_id_or_path, str) and "/" in model_id_or_path:
        cached_path = _resolve_hf_hub_path_to_local_cache(model_id_or_path)
        if cached_path is not None:
            return "hf_hub_cached", cached_path
    
    # If not found, provide helpful error
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    raise ValueError(
        f"Model path `{model_id_or_path}` not found. "
        f"Supported formats:\n"
        f"  1. Local checkpoint: /path/to/checkpoint.pt\n"
        f"  2. Local directory: /path/to/model/\n"
        f"  3. HuggingFace Hub (must be cached): TRI-ML/prismatic-vlms/siglip-224px+7b\n"
        f"\nIf using HF Hub path, ensure model is cached locally at:\n"
        f"  {hf_home}/hub/models--<REPO_ID>/snapshots/\n"
        f"\nNote: Remote downloads are not supported; use offline mode only."
    )


def _load_base_components(
    vision_backbone_id: str,
    llm_backbone_id: str,
    arch_specifier: str,
    image_resize_strategy: str,
    llm_max_length: int,
    load_for_training: bool = False,
) -> tuple:
    """
    Load vision backbone, LLM backbone, and related components (offline mode).
    Returns: (vision_backbone, image_transform, llm_backbone, tokenizer)
    """
    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        vision_backbone_id,
        image_resize_strategy,
    )
    # Load LLM Backbone
    overwatch.info(f"Loading Pretrained LLM [bold]{llm_backbone_id}[/] via HF Transformers (local cache only)")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        llm_backbone_id,
        llm_max_length=llm_max_length,
        inference_mode=not load_for_training,
    )
    return vision_backbone, image_transform, llm_backbone, tokenizer


# === Load Pretrained Model ===
def load(
    vla_cfg: VLAConfig,
    checkpoint_path: Union[str, Path] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> OpenVLA:
    """
    Loads a pretrained OpenVLA model from a local path (offline mode).
    
    This function only supports loading from local disk paths. Remote models from HuggingFace Hub
    are no longer supported to enable offline operation.
    
    Supported input types:
    - Checkpoint file (.pt): Reads vla.json from the checkpoint's parent directory
    - Local directory: Reads vla.json or config.json and checkpoints/latest-checkpoint.pt
    
    Args:
        model_id_or_path: Path to checkpoint file or local directory
        cache_dir: Ignored (kept for backward compatibility). Models must be pre-cached locally.
        load_for_training: Whether to load in training mode (affects inference_mode, weight freezing)
        vla_config_overrides: Dictionary to override VLA config values (e.g., trajectory converter params)
    
    Returns:
        OpenVLA: Always returns an OpenVLA instance with trajectory_converter initialized
    """
    # Initialize variables that will be set in different branches
    checkpoint_pt = None
    model_cfg = {}

    path_type, resolved_path = _detect_model_path_type(
        vla_cfg.base_vlm if not checkpoint_path else checkpoint_path)

    # 没有传入checkpoint path，从vla_cfg构建。大部分情况
    if path_type == "hf_hub_cached":
        # Detect path type and resolve path
        path_type, resolved_path = _detect_model_path_type(vla_cfg.base_vlm)
        model_dir = Path(resolved_path)
        overwatch.info(f"Assuming base model directory. Loading from HF Hub cached model `{checkpoint_path}` at `{model_dir}`")
        
        config_json = model_dir / "config.json"
        checkpoint_pt = model_dir / "checkpoints" / "latest-checkpoint.pt"
        if config_json.exists():
            with open(config_json, "r") as f:
                config_data = json.load(f)
            model_cfg = config_data.get("model", {})
    # 传入checkpoint路径本地load
    elif path_type == "checkpoint": 
        checkpoint_pt = Path(resolved_path)
        overwatch.info(f"Loading from checkpoint file `{checkpoint_pt}`")
        # Validate checkpoint path structure
        assert checkpoint_pt.suffix == ".pt" and checkpoint_pt.parent.name == "checkpoints", \
            f"Invalid checkpoint path; expected .../checkpoints/*.pt, got {checkpoint_pt}"
    # 传入path从本地load
    else:
        run_dir = Path(resolved_path)
        overwatch.info(f"Loading from local directory `{run_dir}`")
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert checkpoint_pt.exists(), f"Missing checkpoint at `{checkpoint_pt}`"
    
    # ===== Load Base Components =====
    # Get model configuration - handle missing model_cfg from training config
    if not model_cfg:
        # When loading from training checkpoint, config.json might not have 'model' field
        # Instead, we need to reconstruct it from base_vlm and vla config
        base_vlm_id = vla_cfg.base_vlm
        
        if base_vlm_id:
            overwatch.info(f"Reconstructing model config from base_vlm: {base_vlm_id}")
            # Try to get model config from ModelConfig registry
            try:
                model_id = base_vlm_id
                if "/" in model_id:
                    # Extract the last part from HF path format
                    model_id = model_id.split("/")[-1]
                
                model_cfg_instance = ModelConfig.get_choice_class(model_id)()
                model_cfg = {
                    "model_id": getattr(model_cfg_instance, "model_id", model_id),
                    "vision_backbone_id": getattr(model_cfg_instance, "vision_backbone_id", None),
                    "llm_backbone_id": getattr(model_cfg_instance, "llm_backbone_id", None),
                    "arch_specifier": getattr(model_cfg_instance, "arch_specifier", "gelu-mlp"),
                    "image_resize_strategy": getattr(model_cfg_instance, "image_resize_strategy", "letterbox"),
                    "llm_max_length": getattr(model_cfg_instance, "llm_max_length", 2048),
                }
            except Exception as e:
                overwatch.warning(f"Could not load model from ModelConfig registry: {e}")
                model_cfg = None
    
        # If still no config, provide helpful error
        if not model_cfg:
            raise ValueError(
                f"Could not determine model configuration from checkpoint. "
                f"The config.json is missing the 'model' field. "
                f"base_vlm was: {base_vlm_id}. "
                f"Please ensure checkpoint was saved with proper model configuration."
            )
    
    overwatch.info(
        f"Found Config =>> Loading [bold blue]{model_cfg.get('model_id', 'unknown')}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.get('vision_backbone_id', 'unknown')}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.get('llm_backbone_id', 'unknown')}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.get('arch_specifier', 'unknown')}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )
    
    # Load base components
    vision_backbone, image_transform, llm_backbone, tokenizer = _load_base_components(
        vision_backbone_id=model_cfg["vision_backbone_id"],
        llm_backbone_id=model_cfg["llm_backbone_id"],
        arch_specifier=model_cfg["arch_specifier"],
        image_resize_strategy=model_cfg["image_resize_strategy"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        load_for_training=load_for_training,
    )
    
    # ===== Create Trajectory Converter =====
    # Get trajectory converter config from VLA config or use defaults
    converter_type = vla_cfg.trajectory_converter_type
    converter_n_bins = vla_cfg.trajectory_n_bins
    converter_n_dims = vla_cfg.trajectory_n_dims

    overwatch.info(
        f"Creating Trajectory Converter [bold]{converter_type}[/] with "
        f"n_bins={converter_n_bins}, n_dims={converter_n_dims}"
    )
    
    if converter_type not in TRAJECTORY_CONVERTER_REGISTRY:
        raise ValueError(
            f"Unknown trajectory converter type `{converter_type}`. "
            f"Available: {list(TRAJECTORY_CONVERTER_REGISTRY.keys())}"
        )
    
    trajectory_converter = TRAJECTORY_CONVERTER_REGISTRY[converter_type](
        tokenizer=tokenizer,
        n_bins=converter_n_bins,
        n_dims=converter_n_dims,
    )
    
    # ===== Load VLM Checkpoint and Initialize OpenVLA =====
    overwatch.info(f"Loading VLM from checkpoint and initializing OpenVLA")
    
    # Create OpenVLA using from_pretrained, which automatically calls __init__ with trajectory_converter
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        trajectory_converter=trajectory_converter,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )
    
    overwatch.info(f"Successfully loaded OpenVLA model with trajectory converter")
    
    # Ensure model is on GPU for inference (required for Flash Attention 2 and optimal performance)
    # For training, FSDP will handle device placement automatically via device_id parameter
    if not load_for_training:
        # Check if model is already on CUDA (e.g., moved by FSDP during training)
        first_param = next(vla.parameters())
        if first_param.device.type != "cuda":
            # Inference mode: move to CUDA if available
            if torch.cuda.is_available():
                vla = vla.to("cuda")
                overwatch.info("Model moved to CUDA for inference")
            else:
                overwatch.warning("CUDA not available; model will run on CPU (Flash Attention 2 will not work)")
        else:
            overwatch.info(f"Model already on device: {first_param.device}")
    
    return vla

