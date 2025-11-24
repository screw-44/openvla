"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
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
def _detect_model_path_type(model_id_or_path: Union[str, Path]) -> tuple[str, Union[str, Path]]:
    """
    Detect the type of model path and return (path_type, resolved_path).
    
    Path types:
    - "checkpoint": Local checkpoint file (.pt)
    - "local_dir": Local directory with config.json and checkpoints/
    - "registry_model": Model ID from GLOBAL_REGISTRY
    - "hf_hub_model": Model ID from HF Hub VLA repository
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
    
    # Check if it's a local directory (convert to string for find_model_in_cache)
    model_id_str = str(model_id_or_path)
    flag, resolved_path = find_model_in_cache(model_id_str)
    if flag:
        return "local_dir", Path(resolved_path)
    
    # Check if it's in GLOBAL_REGISTRY
    if model_id_str in GLOBAL_REGISTRY:
        return "registry_model", model_id_str
    
    # Otherwise assume it's a HF Hub model ID
    return "hf_hub_model", model_id_str


def _load_base_components(
    vision_backbone_id: str,
    llm_backbone_id: str,
    arch_specifier: str,
    image_resize_strategy: str,
    llm_max_length: int,
    hf_token: Optional[str] = None,
    load_for_training: bool = False,
) -> tuple:
    """
    Load vision backbone, LLM backbone, and related components.
    Returns: (vision_backbone, image_transform, llm_backbone, tokenizer)
    """
    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        vision_backbone_id,
        image_resize_strategy,
    )
    # Load LLM Backbone
    overwatch.info(f"Loading Pretrained LLM [bold]{llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        llm_backbone_id,
        llm_max_length=llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )
    return vision_backbone, image_transform, llm_backbone, tokenizer


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    vla_config_overrides: Optional[dict] = None,
) -> OpenVLA:
    """
    Loads a pretrained OpenVLA model from either local disk or the HuggingFace Hub.
    
    This unified function automatically detects the input type and handles loading accordingly:
    - Checkpoint file (.pt): Reads vla.json from the checkpoint's parent directory
    - Local directory: Reads vla.json or falls back to model.json
    - Model ID from registry: Downloads from HF Hub and reads model.json
    - HF Hub VLA ID: Downloads from VLA repository and reads vla.json
    
    Args:
        model_id_or_path: Path to checkpoint/directory or model ID
        hf_token: HuggingFace API token for private models
        cache_dir: Directory to cache downloaded models
        load_for_training: Whether to load in training mode (affects inference_mode, weight freezing)
        vla_config_overrides: Dictionary to override VLA config values (e.g., trajectory converter params)
    
    Returns:
        OpenVLA: Always returns an OpenVLA instance with trajectory_converter initialized
    
    Examples:
        # Load from checkpoint file
        vla = load("/path/to/run/checkpoints/step-1000.pt", load_for_training=True)
        
        # Load from registry model ID
        vla = load("prism-dinosiglip-224px")
        
        # Load with config overrides
        vla = load("model_id", vla_config_overrides={"trajectory_n_bins": 512})
    """
    vla_config_overrides = vla_config_overrides or {}
    
    # Detect path type and resolve path
    path_type, resolved_path = _detect_model_path_type(model_id_or_path)
    
    # ===== Branch 1: Checkpoint File =====
    if path_type == "checkpoint":
        checkpoint_pt = Path(resolved_path)
        overwatch.info(f"Loading from checkpoint file `{checkpoint_pt}`")
        
        # Validate checkpoint path structure
        assert checkpoint_pt.suffix == ".pt" and checkpoint_pt.parent.name == "checkpoints", \
            f"Invalid checkpoint path; expected .../checkpoints/*.pt, got {checkpoint_pt}"
        
        run_dir = checkpoint_pt.parents[1]
        config_json = run_dir / "config.json"
        assert config_json.exists(), f"Missing `config.json` for checkpoint at `{run_dir}`"
        
        # Read config - for checkpoints, prefer vla.json if it exists, otherwise use model.json
        vla_config_path = run_dir / "vla.json"
        model_config_path = run_dir / "config.json"
        
        with open(model_config_path, "r") as f:
            config_data = json.load(f)
        
        # Try to get VLA config, fallback to model config
        if "vla" in config_data:
            vla_cfg = config_data["vla"]
            model_cfg = config_data.get("model", {})
        else:
            # Old-style checkpoint with only model config
            model_cfg = config_data.get("model", config_data)
            vla_cfg = {}
    
    # ===== Branch 2: Local Directory =====
    elif path_type == "local_dir":
        run_dir = Path(resolved_path)
        overwatch.info(f"Loading from local directory `{run_dir}`")
        
        config_json = run_dir / "config.json"
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` at `{run_dir}`"
        assert checkpoint_pt.exists(), f"Missing checkpoint at `{checkpoint_pt}`"
        
        with open(config_json, "r") as f:
            config_data = json.load(f)
        
        if "vla" in config_data:
            vla_cfg = config_data["vla"]
            model_cfg = config_data.get("model", {})
        else:
            model_cfg = config_data.get("model", config_data)
            vla_cfg = {}
    
    # ===== Branch 3: Registry Model ID =====
    elif path_type == "registry_model":
        overwatch.warning("Warning!!!: This shouldn't happen. that load model from [registry_model], as it will download model")
        model_id = str(resolved_path)
        overwatch.info(f"Loading model `{model_id}` from registry")
        
        hf_model_id = GLOBAL_REGISTRY[model_id]["model_id"]
        overwatch.info(f"Downloading from HF Hub: {HF_HUB_REPO}/{hf_model_id}")
        
        with overwatch.local_zero_first():
            config_json = hf_hub_download(
                repo_id=HF_HUB_REPO, 
                filename=f"{hf_model_id}/config.json", 
                cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, 
                filename=f"{hf_model_id}/checkpoints/latest-checkpoint.pt", 
                cache_dir=cache_dir
            )
        
        with open(config_json, "r") as f:
            config_data = json.load(f)
        
        # Registry models use standard model config
        model_cfg = config_data.get("model", config_data)
        vla_cfg = {}
    
    # ===== Branch 4: HF Hub VLA Model ID =====
    else:  # hf_hub_model
        overwatch.warning("Warning!!!: This shouldn't happen. that load model from [hf_hub_model], as it will download model")
        model_id = str(resolved_path)
        overwatch.info(f"Checking HF Hub for VLA model `{model_id}`")
        
        # Try to find model in HF Hub VLA repository
        hf_path = str(Path(VLA_HF_HUB_REPO) / "pretrained" / model_id)
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find model `{model_id}` in HF Hub VLA repository `{VLA_HF_HUB_REPO}`")
        
        # Find the latest checkpoint
        valid_ckpts = sorted(tmpfs.glob(f"{hf_path}/checkpoints/step-*.pt"))
        if not valid_ckpts:
            raise ValueError(f"No checkpoints found in `{hf_path}/checkpoints/`")
        
        target_ckpt = Path(valid_ckpts[-1]).name
        
        overwatch.info(f"Downloading VLA model `{model_id}` with checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, 
                filename=f"pretrained/{model_id}/config.json", 
                cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, 
                filename=f"pretrained/{model_id}/checkpoints/{target_ckpt}", 
                cache_dir=cache_dir
            )
        
        with open(config_json, "r") as f:
            config_data = json.load(f)
        
        # HF Hub models have vla config
        vla_cfg = config_data.get("vla", {})
        model_cfg = config_data.get("model", {})
    
    # ===== Load Base Components =====
    # Get model configuration - ensure it exists
    if not model_cfg:
        if "vla" in config_data:
            # Try to reconstruct model config from VLA base_vlm
            base_vlm_id = vla_cfg.get("base_vlm")
            if base_vlm_id and base_vlm_id in GLOBAL_REGISTRY:
                model_cfg = GLOBAL_REGISTRY[base_vlm_id].get("model_id", {})
        
        if not model_cfg:
            raise ValueError(f"Could not determine model configuration from `{model_id_or_path}`")
    
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
        hf_token=hf_token,
        load_for_training=load_for_training,
    )
    
    # ===== Create Trajectory Converter =====
    # Get trajectory converter config from VLA config or use defaults
    converter_type = vla_config_overrides.get(
        "trajectory_converter_type",
        vla_cfg.get("trajectory_converter_type", "value_textualize")
    )
    converter_n_bins = vla_config_overrides.get(
        "trajectory_n_bins",
        vla_cfg.get("trajectory_n_bins", 256)
    )
    converter_n_dims = vla_config_overrides.get(
        "trajectory_n_dims",
        vla_cfg.get("trajectory_n_dims", 7)
    )
    
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
    return vla

