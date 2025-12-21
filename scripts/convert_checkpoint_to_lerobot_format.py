#!/usr/bin/env python3
"""
Convert VLA checkpoint to LeRobot-compatible format.

This script converts a checkpoint from the training run format to a format that
lerobot-eval can directly load using --policy.path.

The key differences:
1. Ensures config.json has the correct "type": "vla" field
2. Ensures model.safetensors or latest-checkpoint.safetensors exists
3. Creates a minimal but complete lerobot-compatible directory structure

Usage:
    python scripts/convert_checkpoint_to_lerobot_format.py \
        --input_checkpoint ./runs/distilgpt2+b16+x7--1_test_action-chunk \
        --output_dir ./converted_checkpoints/distilgpt2_vla

Or for a specific checkpoint step:
    python scripts/convert_checkpoint_to_lerobot_format.py \
        --input_checkpoint ./runs/distilgpt2+b16+x7--1_test_action-chunk/checkpoints/step-010000-epoch-00-loss=2.4723.safetensors \
        --output_dir ./converted_checkpoints/distilgpt2_step10000
"""

import argparse
import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_checkpoint_for_lerobot(input_path: Path, output_dir: Path):
    """
    Convert checkpoint to lerobot-compatible format.
    
    Args:
        input_path: Path to input checkpoint (can be directory or file)
        output_dir: Path to output directory
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting checkpoint from: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Case 1: Input is a directory (run directory)
    if input_path.is_dir():
        config_file = input_path / "config.json"
        
        # Load existing config
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {input_path}")
        
        with open(config_file, 'r') as f:
            full_config = json.load(f)
        
        logger.info(f"✓ Loaded config from {config_file}")
        
        # Extract only VLAConfig-compatible fields
        # This is critical: lerobot's VLAConfig only accepts these fields
        vla_config = {
            "type": "vla",
            "action_dim": full_config.get("action_dim", 7),
            "action_horizon": full_config.get("action_horizon", 4),
            "observation_horizon": full_config.get("observation_horizon", 2),
            "device": full_config.get("device", "cuda"),
            "dtype": full_config.get("dtype", "bfloat16"),
        }
        
        # Save minimal config to output
        output_config = output_dir / "config.json"
        with open(output_config, 'w') as f:
            json.dump(vla_config, f, indent=2)
        logger.info(f"✓ Saved lerobot-compatible config to {output_config}")
        
        # Find and copy model checkpoint
        checkpoint_dir = input_path / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory in {input_path}")
        
        # Prefer latest-checkpoint.safetensors
        checkpoint_file = checkpoint_dir / "latest-checkpoint.safetensors"
        if not checkpoint_file.exists():
            # Find the last step checkpoint
            safetensor_files = sorted(checkpoint_dir.glob("step-*.safetensors"))
            if safetensor_files:
                checkpoint_file = safetensor_files[-1]
                logger.info(f"Using latest checkpoint: {checkpoint_file.name}")
            else:
                raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint_dir}")
        
        output_model = output_dir / "model.safetensors"
        shutil.copy2(checkpoint_file, output_model)
        logger.info(f"✓ Copied model checkpoint to {output_model}")
        
        # Also copy latest-checkpoint.safetensors for reference
        if checkpoint_file.name != "latest-checkpoint.safetensors":
            latest_link = checkpoint_dir / "latest-checkpoint.safetensors"
            if not latest_link.exists():
                output_latest = output_dir / "latest-checkpoint.safetensors"
                shutil.copy2(checkpoint_file, output_latest)
                logger.info(f"✓ Copied as latest-checkpoint.safetensors")
    
    # Case 2: Input is a safetensors file
    elif input_path.suffix == ".safetensors":
        # We need a config.json, try to find it from parent directory
        possible_config = input_path.parent.parent / "config.json"
        if not possible_config.exists():
            possible_config = input_path.parent / "config.json"
        if not possible_config.exists():
            raise FileNotFoundError(
                f"Cannot find config.json for checkpoint {input_path}\n"
                "Please ensure config.json exists in parent or parent-parent directory"
            )
        
        with open(possible_config, 'r') as f:
            full_config = json.load(f)
        
        logger.info(f"✓ Loaded config from {possible_config}")
        
        # Extract only VLAConfig-compatible fields
        vla_config = {
            "type": "vla",
            "action_dim": full_config.get("action_dim", 7),
            "action_horizon": full_config.get("action_horizon", 4),
            "observation_horizon": full_config.get("observation_horizon", 2),
            "device": full_config.get("device", "cuda"),
            "dtype": full_config.get("dtype", "bfloat16"),
        }
        
        # Save minimal config to output
        output_config = output_dir / "config.json"
        with open(output_config, 'w') as f:
            json.dump(vla_config, f, indent=2)
        logger.info(f"✓ Saved lerobot-compatible config to {output_config}")
        
        # Copy model checkpoint
        output_model = output_dir / "model.safetensors"
        shutil.copy2(input_path, output_model)
        logger.info(f"✓ Copied model checkpoint to {output_model}")
    
    else:
        raise ValueError(f"Input must be a directory or .safetensors file, got: {input_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Conversion complete!")
    logger.info("="*70)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - config.json: {(output_dir / 'config.json').exists()}")
    logger.info(f"  - model.safetensors: {(output_dir / 'model.safetensors').exists()}")
    
    logger.info(f"\nNow you can use it with lerobot-eval:")
    logger.info(f"  lerobot-eval \\")
    logger.info(f"    --policy.path={output_dir} \\")
    logger.info(f"    --env.type=libero \\")
    logger.info(f"    --eval.n_episodes=10")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VLA checkpoint to LeRobot-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        required=True,
        help="Path to input checkpoint (run directory or safetensors file)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory"
    )
    
    args = parser.parse_args()
    
    try:
        convert_checkpoint_for_lerobot(
            Path(args.input_checkpoint),
            Path(args.output_dir)
        )
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
