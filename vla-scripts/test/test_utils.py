"""
test_utils.py

Shared utilities for test suite.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from prismatic.conf.vla import VLAConfig


def get_default_checkpoint_path() -> str:
    """
    Get default checkpoint path for testing.
    
    Returns the path to base+b32+x7--aff_representation_251117-action_chunk checkpoint.
    Asserts that it exists.
    
    Returns:
        Absolute path to checkpoint file.
    
    Raises:
        AssertionError: If checkpoint doesn't exist.
    """
    # Environment variable override (optional)
    env_path = os.environ.get("VLA_CHECKPOINT_PATH")
    if env_path:
        env_path_obj = Path(env_path)
        assert env_path_obj.exists(), f"VLA_CHECKPOINT_PATH does not exist: {env_path}"
        print(f"[TestUtils] Using checkpoint from VLA_CHECKPOINT_PATH: {env_path}")
        return str(env_path_obj.absolute())
    
    # Default checkpoint (MUST exist)
    default_ckpt = (
        Path(__file__).parent.parent.parent / "runs" / 
        "base+b32+x7--aff_representation_251117-action_chunk" / 
        "checkpoints" / "latest-checkpoint.pt"
    )
    
    assert default_ckpt.exists(), (
        f"Default checkpoint not found: {default_ckpt}\n"
        f"Expected checkpoint at: runs/base+b32+x7--aff_representation_251117-action_chunk/checkpoints/\n"
        f"Available checkpoints: {list((default_ckpt.parent).glob('*.pt')) if default_ckpt.parent.exists() else 'checkpoints dir not found'}"
    )
    
    print(f"[TestUtils] Using default checkpoint: {default_ckpt}")
    return str(default_ckpt.absolute())


def load_vla_config_from_checkpoint(checkpoint_path: str) -> VLAConfig:
    """
    Load VLAConfig from checkpoint's config.json file.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory containing checkpoints/
    
    Returns:
        VLAConfig instance populated from config.json
    
    Raises:
        FileNotFoundError: If config.json not found
        ValueError: If required fields missing from config
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Find config.json - could be in checkpoint's parent (run directory) or one level up
    if checkpoint_path.is_file():
        # If checkpoint file given, look in parent directories
        run_dir = checkpoint_path.parent.parent  # checkpoints/step-xxx.pt -> run_dir
    else:
        # If directory given, assume it's the run directory
        run_dir = checkpoint_path
    
    config_path = run_dir / "config.json"
    if not config_path.exists():
        # Try one level up
        config_path = run_dir.parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot find config.json for checkpoint {checkpoint_path}\n"
            f"Searched: {run_dir / 'config.json'}, {run_dir.parent / 'config.json'}"
        )
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Extract VLA config from the JSON
    # Config structure typically has "vla" key with VLA fields
    if "vla" in config_data:
        vla_cfg_dict = config_data["vla"]
    elif "vla_cfg" in config_data:
        vla_cfg_dict = config_data["vla_cfg"]
    elif "vla_id" in config_data:
        # Flat structure with VLA fields directly in root
        vla_cfg_dict = config_data
    else:
        # Try to extract from nested structure
        vla_cfg_dict = config_data
    
    # Required fields for VLAConfig
    required_fields = ["vla_id", "base_vlm"]
    for field in required_fields:
        if field not in vla_cfg_dict:
            raise ValueError(
                f"Missing required VLA config field '{field}' in {config_path}\n"
                f"Available fields: {list(vla_cfg_dict.keys())}"
            )
    
    # Create VLAConfig instance - only include fields that are in VLAConfig
    vla_cfg = VLAConfig(
        vla_id=vla_cfg_dict["vla_id"],
        base_vlm=vla_cfg_dict["base_vlm"],
        freeze_vision_backbone=vla_cfg_dict.get("freeze_vision_backbone", False),
        freeze_llm_backbone=vla_cfg_dict.get("freeze_llm_backbone", False),
        unfreeze_last_llm_layer=vla_cfg_dict.get("unfreeze_last_llm_layer", False),
        shuffle_buffer_size=vla_cfg_dict.get("shuffle_buffer_size", 256_000),
        global_batch_size=vla_cfg_dict.get("global_batch_size", 256),
        per_device_batch_size=vla_cfg_dict.get("per_device_batch_size", 32),
        learning_rate=vla_cfg_dict.get("learning_rate", 2e-5),
        weight_decay=vla_cfg_dict.get("weight_decay", 0.0),
        max_grad_norm=vla_cfg_dict.get("max_grad_norm", 1.0),
        lr_scheduler_type=vla_cfg_dict.get("lr_scheduler_type", "constant"),
        warmup_ratio=vla_cfg_dict.get("warmup_ratio", 0.0),
        train_strategy=vla_cfg_dict.get("train_strategy", "fsdp-full-shard"),
        enable_gradient_checkpointing=vla_cfg_dict.get("enable_gradient_checkpointing", True),
        enable_mixed_precision_training=vla_cfg_dict.get("enable_mixed_precision_training", True),
        reduce_in_full_precision=vla_cfg_dict.get("reduce_in_full_precision", True),
        trajectory_converter_type=vla_cfg_dict.get("trajectory_converter_type", "value_textualize"),
        trajectory_n_bins=vla_cfg_dict.get("trajectory_n_bins", 256),
        trajectory_n_dims=vla_cfg_dict.get("trajectory_n_dims", 7),
    )
    
    return vla_cfg


if __name__ == "__main__":
    # Quick test
    print("Testing checkpoint path...")
    try:
        checkpoint = get_default_checkpoint_path()
        print(f"✓ Found checkpoint: {checkpoint}")
        
        # Test VLAConfig loading
        vla_cfg = load_vla_config_from_checkpoint(checkpoint)
        print(f"✓ Loaded VLAConfig: vla_id={vla_cfg.vla_id}, base_vlm={vla_cfg.base_vlm}")
    except (AssertionError, FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
