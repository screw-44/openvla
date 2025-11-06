"""
pose_tokenizer.py

Extension class for tokenizing 9D poses using the existing vocabulary bins.
Each dimension of the pose is mapped to one of 256 tokens.
"""

from typing import List, Union, Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


class PoseTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        bins: int = 256, 
        min_pose: float = -1, 
        max_pose: float = 3.4,
        pose_dim: int = 9
    ) -> None:
        """
        Discretizes continuous 9D poses into N bins per dimension and maps to the least used tokens.
        
        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_pose: Minimum pose value (for clipping, setting lower bound on bin interval).
        :param max_pose: Maximum pose value (for clipping, setting upper bound on bin interval).
        :param pose_dim: Dimensionality of each pose (default 9D).
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_pose = min_pose
        self.max_pose = max_pose
        self.pose_dim = pose_dim
        
        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_pose, max_pose, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
    
    def __call__(self, poses: np.ndarray) -> str:
        """
        Tokenize poses to string representation.
        
        Args:
            poses: numpy array of shape (num_poses, 9) or (26, 9) for your use case
        
        Returns:
            String representation of tokenized poses
        """
        # Handle single pose vs multiple poses
        if len(poses.shape) == 1:
            poses = poses.reshape(1, -1)
        
        # Ensure correct shape
        assert poses.shape[1] == self.pose_dim, f"Expected {self.pose_dim}D poses, got {poses.shape[1]}D"
        
        # Clip poses to valid range
        poses_clipped = np.clip(poses, a_min=float(self.min_pose), a_max=float(self.max_pose))
        
        # Discretize each dimension
        discretized_poses = np.digitize(poses_clipped, self.bins)
        
        # Convert to token IDs (using the last n_bins tokens of vocabulary)
        token_ids = []
        for pose in discretized_poses:
            for dim_value in pose:
                token_id = self.tokenizer.vocab_size - dim_value
                token_ids.append(token_id)
        
        # Decode to string
        return self.tokenizer.decode(token_ids)

    def encode_actions(self, poses: np.ndarray) -> List[int]:
        """
        Encode poses to token IDs.
        
        Args:
            poses: numpy array of shape (num_poses, 9)
        
        Returns:
            List of token IDs
        """
        # Handle single pose vs multiple poses  
        if len(poses.shape) == 1:
            poses = poses.reshape(1, -1)
        
        # Ensure correct shape
        assert poses.shape[1] == self.pose_dim, f"Expected {self.pose_dim}D poses, got {poses.shape[1]}D"
        
        # Clip poses to valid range
        poses_clipped = np.clip(poses, a_min=float(self.min_pose), a_max=float(self.max_pose))
        
        # Discretize each dimension
        discretized_poses = np.digitize(poses_clipped, self.bins)
        
        # Convert to token IDs
        token_ids = []
        for pose in discretized_poses:
            for dim_value in pose:
                token_id = self.tokenizer.vocab_size - dim_value
                token_ids.append(token_id)
        
        return token_ids
    
    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Decode token IDs back to continuous actions.

        Args:
            token_ids: Array of token IDs
            num_poses: Number of poses expected (e.g., 26 for your case)
        
        Returns:
            numpy array of shape (num_poses, 9)
        """

        # Convert token IDs back to discretized values
        discretized_values = self.tokenizer.vocab_size - token_ids
        discretized_values = np.clip(discretized_values - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        
        # Convert back to continuous values using bin centers
        continuous_values = self.bin_centers[discretized_values]
        
        # Reshape to (num_poses, pose_dim)
        poses = continuous_values.reshape(-1, self.pose_dim)
        
        return poses
    
    # def get_num_tokens_for_poses(self, num_poses: int) -> int:
    #     """Get the number of tokens needed for a given number of poses."""
    #     return num_poses * self.pose_dim
    
    @property
    def vocab_size(self) -> int:
        return self.n_bins