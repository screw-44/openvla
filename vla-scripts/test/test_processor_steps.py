"""
test_processor_steps.py

Unit tests for individual processor steps.
Tests each step in isolation to verify correctness.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from PIL import Image

from prismatic.vla.processor_wrapper import (
    TrajectoryRetrievalProcessorStep,
    TrajectoryCompressionProcessorStep,
    VLATokenizerProcessorStep,
    VLAActionDecoderProcessorStep,
)
from prismatic.vla.tokenizer import ValueTextualizeTC, VlaTokenizer
from prismatic.vla.trajectory_compression import (
    BiningTrajectoryCompression,
    UniformBSplineTrajectoryCompression,
)


class TestTrajectoryCompressionProcessorStep(unittest.TestCase):
    """Test trajectory compression step."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_trajectory = np.random.randn(100, 7)  # 100 timesteps, 7D action
    
    def test_bining_compression(self):
        """Test binning compression method."""
        step = TrajectoryCompressionProcessorStep(
            compression_method="bining",
            target_length=50,
        )
        
        item = {'full_trajectory': self.test_trajectory}
        result = step(item)
        
        # Check output exists
        self.assertIn('compressed_trajectory', result)
        
        # Check shape
        compressed = result['compressed_trajectory']
        self.assertEqual(compressed.shape, (50, 7))
        
        # Check is tensor
        self.assertIsInstance(compressed, torch.Tensor)
    
    def test_bspline_compression(self):
        """Test B-spline compression method."""
        step = TrajectoryCompressionProcessorStep(
            compression_method="uniform_bspline",
            target_length=20,
            degree=3,
        )
        
        item = {'full_trajectory': self.test_trajectory}
        result = step(item)
        
        # Check output exists
        self.assertIn('compressed_trajectory', result)
        
        # Check dimensions
        compressed = result['compressed_trajectory']
        self.assertEqual(compressed.ndim, 2)
        self.assertEqual(compressed.shape[1], 7)  # Action dim preserved
    
    def test_fix_freq_mode(self):
        """Test fix_freq mode with tuple input."""
        step = TrajectoryCompressionProcessorStep(
            compression_method="bining",
            target_length=50,
        )
        
        # Simulate fix_freq mode
        relative_pos = 0.5
        item = {'full_trajectory': (self.test_trajectory, relative_pos)}
        
        result = step(item)
        
        # Should still produce compressed trajectory
        self.assertIn('compressed_trajectory', result)
        compressed = result['compressed_trajectory']
        self.assertEqual(compressed.shape[1], 7)


class TestVLAActionDecoderProcessorStep(unittest.TestCase):
    """Test action decoder step."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load tokenizer from VLA checkpoint (no download)
        from test_utils import get_default_checkpoint_path, load_vla_config_from_checkpoint
        from prismatic.models import load
        
        checkpoint_path = get_default_checkpoint_path()
        vla_cfg = load_vla_config_from_checkpoint(checkpoint_path)
        vla = load(
            vla_cfg=vla_cfg,
            checkpoint_path=checkpoint_path,
            load_for_training=False,
        )
        
        # Use VLA's tokenizer (already loaded from local cache)
        self.base_tokenizer = vla.llm_backbone.tokenizer
        
        self.trajectory_converter = ValueTextualizeTC(
            tokenizer=self.base_tokenizer,
            n_bins=256,
            n_dims=7,
        )
    
    def test_decode_token_ids(self):
        """Test decoding token IDs to continuous actions."""
        step = VLAActionDecoderProcessorStep(self.trajectory_converter)
        
        # Create test trajectory
        test_traj = np.random.uniform(-1, 1, size=(50, 7))
        
        # Encode to tokens
        test_traj_flat = test_traj.T.flatten()  # Flatten as converter expects
        test_traj_clipped = np.clip(test_traj_flat, -1.0, 1.0)
        discretized = np.digitize(test_traj_clipped, self.trajectory_converter.bins)
        token_ids = self.trajectory_converter.tokenizer.vocab_size - discretized
        
        # Decode
        result = step({'action_token_ids': token_ids})
        
        # Check output
        self.assertIn('action', result)
        decoded = result['action']
        
        # Check shape
        self.assertEqual(decoded.shape, (50, 7))
        
        # Check values are in valid range
        self.assertTrue(np.all(decoded >= -1.0))
        self.assertTrue(np.all(decoded <= 1.0))
    
    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        step = VLAActionDecoderProcessorStep(self.trajectory_converter)
        
        # Create torch tensor input
        token_ids = torch.randint(
            low=self.trajectory_converter.trajectory_token_begin_idx,
            high=self.trajectory_converter.tokenizer.vocab_size,
            size=(350,)  # 50 timesteps * 7 dims
        )
        
        result = step({'action_token_ids': token_ids})
        
        # Should convert to numpy internally
        self.assertIn('action', result)
        self.assertIsInstance(result['action'], np.ndarray)


class TestPipelineIntegration(unittest.TestCase):
    """Test that steps can be chained together."""
    
    def test_compression_to_decoder_chain(self):
        """Test chaining compression and decoder steps."""
        from prismatic.vla.processor_wrapper import SimplePipeline
        from transformers import AutoTokenizer
        
        # Setup
        base_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
        )
        
        trajectory_converter = ValueTextualizeTC(
            tokenizer=base_tokenizer,
            n_bins=256,
            n_dims=7,
        )
        
        # Create pipeline
        steps = [
            TrajectoryCompressionProcessorStep("bining", target_length=50),
        ]
        
        pipeline = SimplePipeline(steps, name="test_pipeline")
        
        # Test data
        test_traj = np.random.uniform(-1, 1, size=(100, 7))
        item = {'full_trajectory': test_traj}
        
        # Run pipeline
        result = pipeline(item)
        
        # Check output
        self.assertIn('compressed_trajectory', result)
        self.assertEqual(result['compressed_trajectory'].shape, (50, 7))


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
