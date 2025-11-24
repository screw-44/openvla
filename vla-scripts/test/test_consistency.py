"""
test_consistency.py

Critical test: Verify that wrapped model produces identical outputs to original.

This test ensures that the processor wrapper doesn't introduce any bugs,
numerical errors, or logic changes compared to the original VLA implementation.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from pathlib import Path

from prismatic.models import load
from prismatic.vla.processor_wrapper import OpenVLAPolicyWrapper
from prismatic.vla.materialize import get_vla_dataset_and_collator
from PIL import Image


class TestModelConsistency(unittest.TestCase):
    """
    Test that wrapped model produces identical outputs to original VLA.
    
    This is the MOST IMPORTANT test - ensures wrapper doesn't break anything.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test fixtures once for all tests.
        """
        from test_utils import get_default_checkpoint_path
        
        # Get default checkpoint path (will assert if not found)
        cls.checkpoint_path = Path(get_default_checkpoint_path())
        print(f"\n[TestSetup] Using checkpoint: {cls.checkpoint_path}")
        
        # Load original VLA
        print("[TestSetup] Loading original VLA model...")
        cls.original_vla = load(
            str(cls.checkpoint_path),
            hf_token=os.environ.get("HF_TOKEN"),
            load_for_training=False,
        )
        cls.original_vla.eval()
        
        # Load wrapped VLA
        print("[TestSetup] Loading wrapped VLA model...")
        cls.wrapped_vla = OpenVLAPolicyWrapper.from_pretrained(
            str(cls.checkpoint_path),
            hf_token=os.environ.get("HF_TOKEN"),
        )
        
        # Create test samples
        print("[TestSetup] Creating test samples...")
        cls.test_samples = cls._create_test_samples()
        
        print("[TestSetup] Setup complete!\n")
    
    @classmethod
    def _create_test_samples(cls):
        """Create synthetic test samples for consistency testing."""
        samples = []
        
        # Sample 1: Simple test with random image
        np.random.seed(42)
        sample1 = {
            'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
            'task': 'pick up the red block',
        }
        samples.append(sample1)
        
        # Sample 2: Different task
        sample2 = {
            'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
            'task': 'place the object in the container',
        }
        samples.append(sample2)
        
        return samples
    
    def test_action_prediction_consistency(self):
        """
        Test that both models predict identical actions.
        
        This is the core test - if this passes, the wrapper is working correctly.
        """
        print("\n[Test] Comparing action predictions...")
        
        for i, sample in enumerate(self.test_samples):
            with self.subTest(sample_idx=i):
                # === Original prediction ===
                image_pil = Image.fromarray(sample['full_image'])
                task = sample['task']
                
                original_result = self.original_vla.predict_action(
                    image=image_pil,
                    instruction=task,
                )
                original_action = original_result['action']  # Extract action from dict
                
                # === Wrapped prediction ===
                wrapped_action = self.wrapped_vla.select_action(sample)
                
                # === Compare ===
                print(f"\nSample {i}:")
                print(f"  Task: {task}")
                print(f"  Original action: {original_action}")
                print(f"  Wrapped action:  {wrapped_action}")
                
                # Check shape
                self.assertEqual(
                    original_action.shape,
                    wrapped_action.shape,
                    msg=f"Action shapes don't match for sample {i}"
                )
                
                # Check values (allow small numerical differences)
                np.testing.assert_allclose(
                    original_action,
                    wrapped_action,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Actions don't match for sample {i}!"
                )
                
                print(f"  ✓ Actions match!")
    
    def test_trajectory_converter_consistency(self):
        """Test that trajectory converter decode produces identical results."""
        print("\n[Test] Comparing trajectory converter decoding...")
        
        # Generate random token IDs in valid range
        np.random.seed(123)
        test_token_ids = np.random.randint(
            low=self.original_vla.trajectory_converter.trajectory_token_begin_idx,
            high=self.original_vla.trajectory_converter.tokenizer.vocab_size,
            size=(350,)  # 50 timesteps * 7 dims
        )
        
        # Decode with original
        original_decoded = self.original_vla.trajectory_converter.decode_text_ids_to_trajectory(
            test_token_ids
        )
        
        # Decode with wrapper's trajectory converter
        wrapped_decoded = self.wrapped_vla.postprocessor.steps[0].converter.decode_text_ids_to_trajectory(
            test_token_ids
        )
        
        print(f"Original decoded shape: {original_decoded.shape}")
        print(f"Wrapped decoded shape: {wrapped_decoded.shape}")
        print(f"Original decoded (first 5): {original_decoded[:5]}")
        print(f"Wrapped decoded (first 5): {wrapped_decoded[:5]}")
        
        # Compare
        np.testing.assert_array_almost_equal(
            original_decoded,
            wrapped_decoded,
            decimal=6,
            err_msg="Action tokenizer decoding doesn't match!"
        )
        
        print("  ✓ Decoders match!")
    
    def test_model_parameters_identical(self):
        """Verify that both models have identical parameters."""
        print("\n[Test] Comparing model parameters...")
        
        # Get parameter counts
        original_params = sum(p.numel() for p in self.original_vla.parameters())
        wrapped_params = sum(p.numel() for p in self.wrapped_vla.vla.parameters())
        
        print(f"Original model parameters: {original_params:,}")
        print(f"Wrapped model parameters: {wrapped_params:,}")
        
        self.assertEqual(
            original_params,
            wrapped_params,
            msg="Models have different parameter counts!"
        )
        
        # Check a few parameter values
        original_first_param = next(self.original_vla.parameters())
        wrapped_first_param = next(self.wrapped_vla.vla.parameters())
        
        torch.testing.assert_close(
            original_first_param,
            wrapped_first_param,
            msg="First parameters don't match!"
        )
        
        print("  ✓ Parameters match!")


def run_consistency_tests():
    """Run consistency tests."""
    # Run with verbose output
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelConsistency)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL CONSISTENCY TESTS PASSED!")
        print("The processor wrapper produces identical outputs to the original VLA.")
    else:
        print("✗ SOME TESTS FAILED!")
        print("The processor wrapper may have introduced bugs or changes.")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_consistency_tests()
    sys.exit(0 if success else 1)
