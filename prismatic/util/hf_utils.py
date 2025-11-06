"""
Hugging Face utilities for model cache management and detection.
"""

import os
import glob
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# Set up logging
logger = logging.getLogger(__name__)


def get_hf_cache_dirs() -> List[str]:
    """
    Get possible Hugging Face cache directories.
    
    Returns:
        List of possible cache directory paths
    """
    possible_dirs = []
    
    # Check environment variable
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        possible_dirs.append(os.path.join(hf_home, 'hub'))
    
    # Check default locations
    home_dir = os.path.expanduser('~')
    default_locations = [
        os.path.join(home_dir, '.cache', 'huggingface', 'hub'),
        os.path.join(home_dir, '.cache', 'huggingface', 'transformers'),
        '/root/.cache/huggingface/hub',
        # Add custom locations that might be used in your environment
        '/inspire/ssd/project/robot-decision/hexinyu-253108100063/Software/huggingface/hub',
    ]
    
    possible_dirs.extend(default_locations)
    
    # Filter existing directories and remove duplicates
    existing_dirs = []
    seen = set()
    for d in possible_dirs:
        if os.path.isdir(d) and d not in seen:
            existing_dirs.append(d)
            seen.add(d)
    
    return existing_dirs


def normalize_model_id(model_id: str) -> str:
    """
    Convert model ID to the format used in HF cache directory names.
    
    Args:
        model_id: Model ID like 'bert-base-uncased' or 'facebook/sam-vit-base'
        
    Returns:
        Normalized model ID like 'models--bert-base-uncased' or 'models--facebook--sam-vit-base'
    """
    # Replace '/' with '--' and add 'models--' prefix
    normalized = model_id.replace('/', '--')
    if not normalized.startswith('models--'):
        normalized = f'models--{normalized}'
    
    return normalized


def find_model_in_cache(model_id: str) -> Tuple[bool, str]:
    """
    Find a model in Hugging Face cache directories.
    
    This function searches for the model in the snapshots subdirectory of HF cache.
    It automatically detects HF cache directories and looks for the model.
    
    Args:
        model_id: The model ID to search for (e.g., 'bert-base-uncased', 'facebook/sam-vit-base')
        
    Returns:
        Tuple of (found: bool, path: str)
        - If found: (True, path_to_model_snapshot_dir)
        - If not found: (False, original_model_id)
    """
    
    # Get all possible cache directories
    cache_dirs = get_hf_cache_dirs()
    
    if not cache_dirs:
        logger.warning("No Hugging Face cache directories found")
        return False, model_id
    
    # Normalize the model ID for directory search
    normalized_model_id = normalize_model_id(model_id)
    
    logger.info(f"Searching for model: {model_id} (normalized: {normalized_model_id})")
    logger.info(f"Searching in cache directories: {cache_dirs}")
    
    for cache_dir in cache_dirs:
        try:
            # Look for the model directory
            model_dir = os.path.join(cache_dir, normalized_model_id)
            
            if not os.path.isdir(model_dir):
                logger.debug(f"Model directory not found: {model_dir}")
                continue
            
            # Look for snapshots directory
            snapshots_dir = os.path.join(model_dir, 'snapshots')
            
            if not os.path.isdir(snapshots_dir):
                logger.debug(f"Snapshots directory not found: {snapshots_dir}")
                continue
            
            # Find snapshot subdirectories (should contain hash directories)
            snapshot_subdirs = [d for d in os.listdir(snapshots_dir) 
                              if os.path.isdir(os.path.join(snapshots_dir, d))]
            
            if not snapshot_subdirs:
                logger.debug(f"No snapshot subdirectories found in: {snapshots_dir}")
                continue
            
            # Return the first (and usually only) snapshot directory
            snapshot_path = os.path.join(snapshots_dir, snapshot_subdirs[0])
            
            logger.info(f"Found model at: {snapshot_path}")
            return True, snapshot_path
            
        except Exception as e:
            logger.error(f"Error searching in {cache_dir}: {e}")
            continue
    
    logger.warning(f"Model {model_id} not found in any cache directory")
    return False, model_id


def find_all_cached_models() -> List[Tuple[str, str]]:
    """
    Find all cached models in Hugging Face cache directories.
    
    Returns:
        List of tuples (model_name, snapshot_path)
    """
    cached_models = []
    cache_dirs = get_hf_cache_dirs()
    
    for cache_dir in cache_dirs:
        try:
            # Find all models-- directories
            model_pattern = os.path.join(cache_dir, 'models--*')
            model_dirs = glob.glob(model_pattern)
            
            for model_dir in model_dirs:
                if not os.path.isdir(model_dir):
                    continue
                
                # Extract model name from directory
                model_name = os.path.basename(model_dir)
                if model_name.startswith('models--'):
                    # Convert back to original format
                    clean_name = model_name[8:]  # Remove 'models--' prefix
                    clean_name = clean_name.replace('--', '/')
                    
                    # Check for snapshots
                    snapshots_dir = os.path.join(model_dir, 'snapshots')
                    if os.path.isdir(snapshots_dir):
                        snapshot_subdirs = [d for d in os.listdir(snapshots_dir) 
                                          if os.path.isdir(os.path.join(snapshots_dir, d))]
                        
                        if snapshot_subdirs:
                            snapshot_path = os.path.join(snapshots_dir, snapshot_subdirs[0])
                            cached_models.append((clean_name, snapshot_path))
        
        except Exception as e:
            logger.error(f"Error scanning cache directory {cache_dir}: {e}")
    
    return cached_models


def get_model_path_or_id(model_id: str, prefer_cache: bool = True) -> str:
    """
    Get the local path to a cached model if available, otherwise return the model ID.
    
    Args:
        model_id: The model ID to search for
        prefer_cache: If True, prefer local cache over downloading
        
    Returns:
        Local path if found in cache, otherwise the original model_id
    """
    if prefer_cache:
        found, path = find_model_in_cache(model_id)
        if found:
            return path
    
    return model_id


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        found, path = find_model_in_cache(model_id)
        print(f"Model: {model_id}")
        print(f"Found: {found}")
        print(f"Path: {path}")
    else:
        # Test with some common models
        test_models = [
            'bert-base-uncased',
            'facebook/sam-vit-base',
            'lmsys/vicuna-7b-v1.5',
            'TRI-ML/prismatic-vlms',
            'timm/ViT-SO400M-14-SigLIP'
        ]
        
        print("Testing model search:")
        print("=" * 50)
        
        for model in test_models:
            found, path = find_model_in_cache(model)
            status = "✓ FOUND" if found else "✗ NOT FOUND"
            print(f"{status:<12} {model:<30} -> {path}")
        
        print("\n" + "=" * 50)
        print("All cached models:")
        cached_models = find_all_cached_models()
        for model_name, snapshot_path in cached_models:
            print(f"  {model_name:<30} -> {snapshot_path}")
