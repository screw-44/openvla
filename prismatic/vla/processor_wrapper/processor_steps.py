"""
processor_steps.py

Individual processor steps that can be chained to form preprocessing/postprocessing pipelines.
Each step encapsulates existing VLA logic without modification.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.tokenizer import VlaTokenizer, BaseTrajectoryConverter
from prismatic.vla.trajectory_compression import TRAJECTORY_COMPRESSION_REGISTRY

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


class TrajectoryRetrievalProcessorStep:
    """
    Retrieves full trajectory for an episode frame.
    
    Replicates the logic from MyLeRobotDataset.get_trajectory_for_item:
    - Accesses episode-level data from dataset
    - Supports both 'positional' and 'fix_freq' modes
    - Returns complete trajectory from current frame to episode end
    """
    
    def __init__(
        self,
        dataset_ref,
        exp_type: str = "positional",
    ):
        """
        Args:
            dataset_ref: Reference to MyLeRobotDataset instance
            exp_type: Experiment type ('positional' or 'fix_freq')
        """
        self.dataset = dataset_ref
        self.exp_type = exp_type
    
    def __call__(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve full trajectory for the given item.
        
        Args:
            item_dict: Dictionary from LeRobotDataset.__getitem__ containing:
                - 'episode_index': episode ID
                - 'frame_index': frame ID within episode
                - 'action': single-frame action (ignored)
                - other observation fields
        
        Returns:
            item_dict with added 'full_trajectory' field
        """
        # Extract episode and frame indices
        episode_id = item_dict['episode_index'].item() if torch.is_tensor(item_dict['episode_index']) else item_dict['episode_index']
        frame_id = item_dict['frame_index'].item() if torch.is_tensor(item_dict['frame_index']) else item_dict['frame_index']
        
        # Get episode boundaries from metadata
        episode_from_id = self.dataset.dataset.meta.episodes['dataset_from_index'][episode_id]
        episode_to_id = self.dataset.dataset.meta.episodes['dataset_to_index'][episode_id]
        
        # Retrieve trajectory based on exp_type
        if self.exp_type == "positional":
            # From current frame to episode end
            original_trajectory = np.array(
                self.dataset.dataset.hf_dataset['action'][episode_from_id + frame_id:episode_to_id]
            )
            item_dict['full_trajectory'] = original_trajectory
            
        elif self.exp_type == "fix_freq":
            # Complete episode trajectory + relative position
            complete_trajectory = np.array(
                self.dataset.dataset.hf_dataset['action'][episode_from_id:episode_to_id]
            )
            relative_pos = frame_id / (episode_to_id - episode_from_id)
            item_dict['full_trajectory'] = (complete_trajectory, relative_pos)
        
        else:
            raise ValueError(f"Unknown exp_type: {self.exp_type}")
        
        return item_dict


class TrajectoryCompressionProcessorStep:
    """
    Compresses trajectory using specified compression method.
    
    Uses existing TRAJECTORY_COMPRESSION_REGISTRY to apply compression:
    - BSpline compression
    - Binning
    - Action chunking
    - etc.
    """
    
    def __init__(self, compression_method: str, **compression_kwargs):
        """
        Args:
            compression_method: Name of compression method in TRAJECTORY_COMPRESSION_REGISTRY
            compression_kwargs: Additional kwargs for compression method
        """
        self.compression_method = compression_method
        
        # Initialize compressor from registry
        if compression_method not in TRAJECTORY_COMPRESSION_REGISTRY:
            raise ValueError(
                f"Unknown compression method: {compression_method}. "
                f"Available: {list(TRAJECTORY_COMPRESSION_REGISTRY.keys())}"
            )
        
        self.compressor = TRAJECTORY_COMPRESSION_REGISTRY[compression_method](**compression_kwargs)
    
    def __call__(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress the full trajectory.
        
        Args:
            item_dict: Dictionary with 'full_trajectory' field
        
        Returns:
            item_dict with added 'compressed_trajectory' field
        """
        full_traj = item_dict['full_trajectory']
        
        # Handle different exp_types
        if isinstance(full_traj, tuple):
            # fix_freq mode: (complete_trajectory, relative_pos)
            complete_traj, relative_pos = full_traj
            
            # Check if compressor supports fix_freq mode
            if hasattr(self.compressor, 'exp_type') and self.compressor.exp_type == "fix_freq":
                compressed = self.compressor(complete_traj, relative_pos)
            else:
                # Fallback: compress from current position
                current_idx = int(relative_pos * len(complete_traj))
                compressed = self.compressor(complete_traj[current_idx:])
        else:
            # positional mode: direct compression
            compressed = self.compressor(full_traj)
        
        # Convert to tensor
        item_dict['compressed_trajectory'] = torch.tensor(compressed, dtype=torch.float32)
        
        return item_dict


class VLATokenizerProcessorStep:
    """
    Tokenizes observations and trajectory into model inputs.
    
    Wraps existing VlaTokenizer.tokenize_batch logic:
    - Builds conversation prompt
    - Encodes language with base tokenizer
    - Discretizes trajectory with trajectory_converter
    - Transforms image
    """
    
    def __init__(self, vla_tokenizer: VlaTokenizer):
        """
        Args:
            vla_tokenizer: Initialized VlaTokenizer instance
        """
        self.tokenizer = vla_tokenizer
    
    def __call__(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize item into model inputs.
        
        Args:
            item_dict: Dictionary with:
                - 'observation.images.xxx': image tensor
                - 'task': task description string
                - 'compressed_trajectory': compressed action trajectory
                - 'dataset_names': dataset identifier
        
        Returns:
            item_dict with added fields:
                - 'input_ids': tokenized prompt
                - 'labels': labels for training (action tokens only)
                - 'pixel_values': preprocessed image
                - 'attention_mask': attention mask
        """
        # Extract required fields
        # Find image key (could be various formats)
        image_key = None
        for key in item_dict.keys():
            if 'image' in key.lower() and 'observation' in key:
                image_key = key
                break
        
        if image_key is None:
            # Try alternative formats
            if 'rgb' in item_dict:
                image_key = 'rgb'
            else:
                raise ValueError(f"No image found in item_dict keys: {list(item_dict.keys())}")
        
        # Construct batch for tokenizer (single sample)
        batch = {
            'rgb': item_dict[image_key],
            'language': item_dict.get('task', item_dict.get('instruction', 'perform task')),
            'trajectory': item_dict['compressed_trajectory'],
            'dataset_names': item_dict.get('dataset_names', 'unknown'),
        }
        
        # Call existing tokenize_batch method
        tokenized = self.tokenizer.tokenize_batch(batch, train=True)
        
        # Merge tokenized fields back into item_dict
        item_dict.update(tokenized)
        
        return item_dict


class VLAActionDecoderProcessorStep:
    """
    Decodes action token IDs back to continuous actions.
    
    Uses existing BaseTrajectoryConverter.decode_text_ids_to_trajectory:
    - Maps token IDs to bin indices
    - Looks up bin centers
    - Reshapes to trajectory format
    """
    
    def __init__(self, trajectory_converter: BaseTrajectoryConverter):
        """
        Args:
            trajectory_converter: Initialized trajectory converter (e.g., ValueTextualizeTC)
        """
        self.converter = trajectory_converter
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode action token IDs to continuous actions.
        
        Args:
            data_dict: Dictionary with 'action_token_ids' field
        
        Returns:
            data_dict with added 'action' field (continuous actions)
        """
        token_ids = data_dict['action_token_ids']
        
        # Ensure numpy array
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().numpy()
        
        # Decode using existing converter
        continuous_actions = self.converter.decode_text_ids_to_trajectory(token_ids)
        
        data_dict['action'] = continuous_actions
        
        return data_dict


class ImagePreprocessorStep:
    """
    Preprocesses images using VLA's image transform.
    Optional step if image preprocessing is needed separately.
    """
    
    def __init__(self, image_transform):
        """
        Args:
            image_transform: Image transform from vision backbone
        """
        self.transform = image_transform
    
    def __call__(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply image transform."""
        # Find image key
        for key in item_dict.keys():
            if 'image' in key.lower():
                img = item_dict[key]
                
                # Apply transform
                if hasattr(self.transform, '__call__'):
                    item_dict[key] = self.transform(img)
                
                break
        
        return item_dict
