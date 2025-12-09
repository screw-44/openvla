"""
eval_with_wrapper.py

Example script showing how to use OpenVLAPolicyWrapper for evaluation.
Can be adapted for lerobot_eval or custom evaluation loops.


python eval_with_wrapper.py  \
    --checkpoint /inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/runs/base+b32+x7--aff_representation_251117-action_chunk/checkpoints/latest-checkpoint.pt \
    --mode libero
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import imageio
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prismatic.vla.processor_wrapper import OpenVLAPolicyWrapper
from prismatic.overwatch import initialize_overwatch

# Initialize logging
overwatch = initialize_overwatch(__name__)

# Setup date/time for logging
DATE = datetime.now().strftime("%Y-%m-%d")
DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_rollout_video(rollout_images, episode_idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task_description}.mp4"
    
    if len(rollout_images) == 0:
        overwatch.warning(f"No images to save for episode {episode_idx}")
        return None
    
    try:
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in rollout_images:
            video_writer.append_data(img)
        video_writer.close()
        overwatch.info(f"Saved rollout MP4 at path {mp4_path}")
        if log_file is not None:
            log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
        return mp4_path
    except Exception as e:
        overwatch.error(f"Failed to save video: {e}")
        return None


def evaluate_policy_in_libero(
    checkpoint_path: str,
    task_suite_name: str = "libero_spatial",
    num_episodes: int = 10,
    max_steps: int = 300,
    save_videos: bool = True,
):
    """
    Evaluate wrapped policy in LIBERO environment.
    
    Args:
        checkpoint_path: Path to VLA checkpoint
        task_suite_name: LIBERO task suite name
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
        save_videos: Whether to save episode videos
    
    Returns:
        success_rate: Overall success rate across episodes
    """
    overwatch.info("="*70)
    overwatch.info("OpenVLA Policy Wrapper Evaluation in LIBERO")
    overwatch.info("="*70)
    
    # Load policy
    overwatch.info(f"Loading policy from: {checkpoint_path}")
    policy = OpenVLAPolicyWrapper.from_pretrained(checkpoint_path)
    
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
    
    try:
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
                
                # Initialize image collection for video
                replay_images = []
                
                # Episode loop
                done = False
                step = 0
                
                while not done and step < max_steps:
                    # Get image observation
                    img = get_libero_image(obs, resize_size=224)
                    
                    # Save image for video replay
                    if save_videos:
                        replay_images.append(img)
                    
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
                
                # Save video of episode
                if save_videos:
                    save_rollout_video(
                        replay_images,
                        episode_idx + 1,
                        success=done and info.get('success', False),
                        task_description=task_description,
                    )
                
                total_episodes += 1
            
            overwatch.info(f"Task {task_id} success rate: {task_successes}/{min(num_episodes, len(initial_states))}")
            total_successes += task_successes
            
            # Close environment for this task
            if hasattr(env, 'close'):
                env.close()
    
    finally:
        # Ensure cleanup happens
        pass
    
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
    policy = OpenVLAPolicyWrapper.from_pretrained(checkpoint_path)
    
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
    parser.add_argument(
        "--save-videos",
        type=bool,
        default=True,
        help="Whether to save episode videos (for libero mode)"
    )
    
    args = parser.parse_args()
    
    # Run selected mode
    if args.mode == "libero":
        evaluate_policy_in_libero(
            checkpoint_path=args.checkpoint,
            task_suite_name=args.task_suite,
            num_episodes=args.num_episodes,
            save_videos=args.save_videos,
        )
    elif args.mode == "demo":
        simple_rollout_demo(
            checkpoint_path=args.checkpoint,
            num_steps=args.num_steps,
        )

