"""
test_integration.py

Integration tests for complete workflows:
- Mini training loop
- Checkpoint save/load
- Environment interaction simulation
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from pathlib import Path

from prismatic.vla.processor_wrapper import OpenVLAPolicyWrapper


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        from test_utils import get_default_checkpoint_path
        
        # Get default checkpoint path
        cls.checkpoint_path = Path(get_default_checkpoint_path())
        print(f"\n[TestSetup] Using checkpoint: {cls.checkpoint_path}")
    
    def test_policy_wrapper_initialization(self):
        """Test that policy wrapper can be initialized."""
        # Use checkpoint from setUpClass
        checkpoint_path = self.checkpoint_path
        
        # Try to initialize
        try:
            policy = OpenVLAPolicyWrapper.from_pretrained(
                str(checkpoint_path),
            )
            
            # Check basic attributes
            self.assertIsNotNone(policy.vla)
            self.assertIsNotNone(policy.preprocessor)
            self.assertIsNotNone(policy.postprocessor)
            
            print("✓ Policy wrapper initialized successfully")
            
        except Exception as e:
            self.fail(f"Failed to initialize policy wrapper: {e}")
    
    def test_select_action_interface(self):
        """Test that select_action interface works."""
        # Use checkpoint from setUpClass
        checkpoint_path = self.checkpoint_path
        
        # Initialize policy
        policy = OpenVLAPolicyWrapper.from_pretrained(
            str(checkpoint_path),
        )
        
        # Create mock observation
        observation = {
            'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
            'task': 'pick up the object',
        }
        
        # Call select_action
        try:
            action = policy.select_action(observation)
            
            # Check output
            self.assertIsInstance(action, np.ndarray)
            self.assertEqual(len(action.shape), 1)  # Should be 1D
            self.assertEqual(action.shape[0], 7)  # 7D action for manipulation
            
            print(f"✓ select_action returned valid action: {action}")
            
        except Exception as e:
            self.fail(f"select_action failed: {e}")
    
    def test_multiple_action_calls(self):
        """Test that multiple action calls work (simulating rollout)."""
        # Use checkpoint from setUpClass
        checkpoint_path = self.checkpoint_path
        
        # Initialize policy
        policy = OpenVLAPolicyWrapper.from_pretrained(
            str(checkpoint_path),
        )
        
        # Simulate multiple timesteps
        num_steps = 5
        actions = []
        
        for step in range(num_steps):
            observation = {
                'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                'task': 'pick up the object',
            }
            
            action = policy.select_action(observation)
            actions.append(action)
        
        # Check all actions are valid
        self.assertEqual(len(actions), num_steps)
        
        for i, action in enumerate(actions):
            self.assertEqual(action.shape, (7,))
        
        print(f"✓ Successfully generated {num_steps} actions")
    
    def test_different_tasks(self):
        """Test with different task descriptions."""
        # Use checkpoint from setUpClass
        checkpoint_path = self.checkpoint_path
        
        # Initialize policy
        policy = OpenVLAPolicyWrapper.from_pretrained(
            str(checkpoint_path),
        )
        
        # Test different tasks
        tasks = [
            'pick up the red block',
            'place the object in the container',
            'push the button',
            'open the drawer',
        ]
        
        for task in tasks:
            observation = {
                'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                'task': task,
            }
            
            try:
                action = policy.select_action(observation)
                self.assertEqual(action.shape, (7,))
                print(f"  ✓ Task '{task[:30]}...' -> action shape {action.shape}")
            except Exception as e:
                self.fail(f"Failed for task '{task}': {e}")


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_integration_tests()
