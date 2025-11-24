"""
eval_with_wrapper.py

Example script showing how to use OpenVLAPolicyWrapper for evaluation.
Can be adapted for lerobot_eval or custom evaluation loops.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prismatic.vla.processor_wrapper import OpenVLAPolicyWrapper
from prismatic.overwatch import initialize_overwatch

# Initialize logging
overwatch = initialize_overwatch(__name__)


def evaluate_policy_in_libero(
    checkpoint_path: str,
    task_suite_name: str = "libero_spatial",
    num_episodes: int = 10,
    max_steps: int = 300,
    device: str = "cuda",
):
    """
    Evaluate wrapped policy in LIBERO environment.
    
    Args:
        checkpoint_path: Path to VLA checkpoint
        task_suite_name: LIBERO task suite name
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
        device: Device to run on
    
    Returns:
        success_rate: Overall success rate across episodes
    """
    overwatch.info("="*70)
    overwatch.info("OpenVLA Policy Wrapper Evaluation in LIBERO")
    overwatch.info("="*70)
    
    # Load policy
    overwatch.info(f"Loading policy from: {checkpoint_path}")
    policy = OpenVLAPolicyWrapper.from_pretrained(
        checkpoint_path,
        device=device,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    
    # Import LIBERO utilities
    # Note: Update these imports based on your actual LIBERO setup
    try:
        from experiments.robot.libero.libero_utils import (
            get_libero_env,
            get_libero_image,
            get_libero_dummy_action,
        )
        from libero.libero import benchmark
    except ImportError:
        overwatch.error("LIBERO not found! Install it or update import paths.")
        return None
    
    # Initialize LIBERO task suite
    overwatch.info(f"Initializing task suite: {task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    # Track results
    total_successes = 0
    total_episodes = 0
    
    # Iterate over tasks
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize environment
        env, task_description = get_libero_env(
            task,
            model_family="openvla",
            resolution=256,
        )
        
        overwatch.info(f"\nTask {task_id}: {task_description}")
        
        # Run episodes for this task
        task_successes = 0
        
        for episode_idx in range(min(num_episodes, len(initial_states))):
            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Episode loop
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get image observation
                img = get_libero_image(obs, resize_size=224)
                
                # Prepare observation dict for policy
                observation = {
                    'full_image': img,
                    'task': task_description,
                }
                
                # Get action from policy
                try:
                    action = policy.select_action(observation)
                except Exception as e:
                    overwatch.error(f"Error in select_action: {e}")
                    break
                
                # Step environment
                obs, reward, done, info = env.step(action)
                step += 1
                
                # Check success
                if done and info.get('success', False):
                    task_successes += 1
                    overwatch.info(f"  Episode {episode_idx+1}: SUCCESS (steps={step})")
                    break
            
            if not done or not info.get('success', False):
                overwatch.info(f"  Episode {episode_idx+1}: FAILURE (steps={step})")
            
            total_episodes += 1
        
        overwatch.info(f"Task {task_id} success rate: {task_successes}/{min(num_episodes, len(initial_states))}")
        total_successes += task_successes
    
    # Calculate overall success rate
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    
    overwatch.info("="*70)
    overwatch.info(f"Overall Success Rate: {success_rate:.2%} ({total_successes}/{total_episodes})")
    overwatch.info("="*70)
    
    return success_rate


def simple_rollout_demo(checkpoint_path: str, num_steps: int = 10):
    """
    Simple demo showing policy rollout (without environment).
    
    Useful for quick testing without needing full LIBERO setup.
    """
    overwatch.info("="*70)
    overwatch.info("Simple Rollout Demo (No Environment)")
    overwatch.info("="*70)
    
    # Load policy
    overwatch.info(f"Loading policy from: {checkpoint_path}")
    policy = OpenVLAPolicyWrapper.from_pretrained(
        checkpoint_path,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    
    # Generate random observations and get actions
    overwatch.info(f"\nGenerating {num_steps} actions...")
    
    for step in range(num_steps):
        # Create random observation
        observation = {
            'full_image': np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
            'task': 'pick up the red block',
        }
        
        # Get action
        action = policy.select_action(observation)
        
        overwatch.info(f"  Step {step+1}: action = {action}")
    
    overwatch.info("\n✓ Demo complete!")


def compare_with_original(checkpoint_path: str):
    """
    Compare wrapped policy with original VLA implementation.
    
    This is essentially a runtime version of test_consistency.py
    """
    overwatch.info("="*70)
    overwatch.info("Comparing Wrapped vs Original VLA")
    overwatch.info("="*70)
    
    from prismatic.models import load
    
    # Load both versions
    overwatch.info("Loading original VLA...")
    original_vla = load(
        checkpoint_path,
        hf_token=os.environ.get("HF_TOKEN"),
        load_for_training=False,
    )
    
    overwatch.info("Loading wrapped VLA...")
    wrapped_vla = OpenVLAPolicyWrapper.from_pretrained(
        checkpoint_path,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    
    # Test on random observations
    num_tests = 5
    overwatch.info(f"\nRunning {num_tests} comparison tests...")
    
    all_match = True
    
    for i in range(num_tests):
        # Create observation
        img_array = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        task = f"test task {i}"
        
        # Original prediction
        img_pil = Image.fromarray(img_array)
        original_result = original_vla.predict_action(
            image=img_pil,
            instruction=task,
        )
        original_action = original_result['action']  # Extract action from dict
        
        # Wrapped prediction
        observation = {
            'full_image': img_array,
            'task': task,
        }
        wrapped_action = wrapped_vla.select_action(observation)
        
        # Compare
        match = np.allclose(original_action, wrapped_action, rtol=1e-5, atol=1e-6)
        
        if match:
            overwatch.info(f"  Test {i+1}: ✓ Actions match!")
        else:
            overwatch.error(f"  Test {i+1}: ✗ Actions differ!")
            overwatch.error(f"    Original: {original_action}")
            overwatch.error(f"    Wrapped:  {wrapped_action}")
            all_match = False
    
    overwatch.info("="*70)
    if all_match:
        overwatch.info("✓ All tests passed! Wrapper is consistent with original.")
    else:
        overwatch.error("✗ Some tests failed! Wrapper may have bugs.")
    overwatch.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA with processor wrapper")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VLA checkpoint file or run directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["libero", "demo", "compare"],
        default="demo",
        help="Evaluation mode: libero (full eval), demo (simple test), compare (consistency check)"
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        help="LIBERO task suite name (for libero mode)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per task (for libero mode)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of steps (for demo mode)"
    )
    
    args = parser.parse_args()
    
    # Run selected mode
    if args.mode == "libero":
        evaluate_policy_in_libero(
            checkpoint_path=args.checkpoint,
            task_suite_name=args.task_suite,
            num_episodes=args.num_episodes,
        )
    elif args.mode == "demo":
        simple_rollout_demo(
            checkpoint_path=args.checkpoint,
            num_steps=args.num_steps,
        )
    elif args.mode == "compare":
        compare_with_original(
            checkpoint_path=args.checkpoint,
        )
