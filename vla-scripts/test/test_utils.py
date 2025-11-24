"""
test_utils.py

Shared utilities for test suite.
"""

import os
from pathlib import Path


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


if __name__ == "__main__":
    # Quick test
    print("Testing checkpoint path...")
    try:
        checkpoint = get_default_checkpoint_path()
        print(f"✓ Found checkpoint: {checkpoint}")
    except AssertionError as e:
        print(f"✗ Error: {e}")
