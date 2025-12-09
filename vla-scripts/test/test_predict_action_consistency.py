"""
test_predict_action_consistency.py

Compare predict_action() output with forward()-based action prediction using REAL dataset samples.
This test verifies that both methods produce identical action outputs from the same dataset inputs.

The key insight:
- predict_action() takes a PIL.Image + instruction string as input
- forward() takes tokenized batch data (input_ids, labels, pixel_values, attention_mask)
- Both should produce the same action when given the same underlying data

Usage:
    python test_predict_action_consistency.py \
        --checkpoint /path/to/checkpoint.pt \
        --dataset-repo-id HuggingFaceVLA/libero \
        --task-ids 0 1 2 \
        --device cuda:0
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PIL import Image

# Import VLA utilities
from prismatic.overwatch import initialize_overwatch
from prismatic.vla import get_vla_dataset_and_collator
from eval_with_wrapper import OpenVLAPolicyWrapper

overwatch = initialize_overwatch(__name__)


class DatasetActionConsistencyTester:
    """Test predict_action() vs forward()-based predictions using real dataset samples."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_repo_id: str = "HuggingFaceVLA/libero",
        task_ids: list = None,
        device: str = "cuda:0",
    ):
        """Initialize VLA model and dataset."""
        self.device = torch.device(device)
        
        # Load VLA model
        overwatch.info(f"Loading VLA model from checkpoint: {checkpoint_path}")
        policy_wrapper = OpenVLAPolicyWrapper.from_pretrained(checkpoint_path)
        self.vla = policy_wrapper.vla
        self.vla = self.vla.to(self.device)
        self.vla.eval()
        
        # Load dataset and collator
        overwatch.info(f"Loading dataset: {dataset_repo_id}")
        if task_ids is None:
            task_ids = [0]
        
        self.vla_dataset, self.trajectory_converter, self.collator = get_vla_dataset_and_collator(
            data_repo_id=dataset_repo_id,
            data_task_ids=task_ids,
            trajectory_compression_method="action_chunk",  # Match training config
            trajectory_converter_type="value_textualize",
            trajectory_n_bins=256,
            trajectory_n_dims=7,
            base_tokenizer=self.vla.llm_backbone.get_tokenizer(),
            prompt_builder_fn=self.vla.llm_backbone.prompt_builder_fn,
            image_transform=self.vla.vision_backbone.get_image_transform(),
            default_image_resolution=self.vla.vision_backbone.default_image_resolution,
        )
        
        overwatch.info(f"Dataset loaded with {len(self.vla_dataset)} samples")
        
        self.results = {
            "predict_action": {},
            "forward_based": {},
            "comparison": {},
        }
    
    def get_dataset_sample(self, index: int = 0) -> Tuple[Image.Image, str, Dict]:
        """
        Get a real dataset sample and convert to required formats.
        
        Returns:
            (image, instruction, batch_dict)
            - image: PIL Image
            - instruction: task instruction string
            - batch_dict: tokenized batch data with input_ids, labels, pixel_values, attention_mask
        """
        # Get raw sample from dataset
        raw_sample = self.vla_dataset[index]
        
        # raw_sample contains: input_ids, labels, pixel_values (all as tensors)
        # We need to reconstruct the original image and instruction
        
        # Extract image from pixel_values (need to denormalize and convert back to PIL)
        pixel_values = raw_sample["pixel_values"]  # Shape: [3, 224, 224] or dict
        
        if isinstance(pixel_values, torch.Tensor):
            # Convert from torch tensor back to PIL Image
            # Assuming it's normalized to [-1, 1] or [0, 1]
            img_array = pixel_values.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            # Denormalize if needed (assuming -1 to 1 range based on SigLIP)
            img_array = ((img_array + 1) / 2 * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")
        
        # For instruction, we'll use a generic one since it's not stored in the sample
        # In real training, this comes from the dataset metadata
        instruction = "complete the task"
        
        # Create batch (add batch dimension)
        batch_dict = {
            "input_ids": raw_sample["input_ids"].unsqueeze(0),  # [1, seq_len]
            "labels": raw_sample["labels"].unsqueeze(0),  # [1, seq_len]
            "pixel_values": raw_sample["pixel_values"].unsqueeze(0),  # [1, 3, H, W]
            "attention_mask": torch.ones_like(raw_sample["input_ids"]).unsqueeze(0),  # [1, seq_len]
        }
        
        return image, instruction, batch_dict
    
    @torch.inference_mode()
    def test_predict_action(self, image: Image.Image, instruction: str, batch_dict: Dict = None) -> Dict[str, Any]:
        """
        Test Method 1: Direct predict_action() call.
        
        If batch_dict is provided, we also log what the expected input_ids and labels should be
        for debugging purposes.
        """
        overwatch.info("=" * 80)
        overwatch.info("TEST 1: predict_action()")
        overwatch.info("=" * 80)
        
        try:
            output = self.vla.predict_action(image, instruction)
            
            result = {
                "status": "success",
                "action_shape": output["action"].shape,
                "action_dtype": str(output["action"].dtype),
                "action_sample": output["action"][:3].tolist(),
                "action_tokens_shape": output["action_tokens"].shape,
                "normalized_actions_shape": output["normalized_actions"].shape,
            }
            
            # Debug: show what batch_dict contains if provided
            if batch_dict is not None:
                overwatch.debug(f"Expected input_ids shape from batch: {batch_dict['input_ids'].shape}")
                overwatch.debug(f"Expected labels shape from batch: {batch_dict['labels'].shape}")
                overwatch.debug(f"Expected labels (first 20 values): {batch_dict['labels'][0, :20]}")
                # Count how many valid labels (not IGNORE_INDEX which is -100)
                valid_label_count = (batch_dict['labels'] != -100).sum().item()
                overwatch.debug(f"Valid labels count (!= -100): {valid_label_count}")
            
            overwatch.info(f"✓ Action shape: {result['action_shape']}")
            overwatch.info(f"✓ Action tokens shape: {result['action_tokens_shape']}")
            overwatch.info(f"✓ Normalized actions shape: {result['normalized_actions_shape']}")
            
            self.results["predict_action"]["output"] = output
            self.results["predict_action"]["summary"] = result
            
            return result
            
        except Exception as e:
            overwatch.error(f"✗ predict_action() failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "status": "failed",
                "error": str(e),
            }
            self.results["predict_action"]["summary"] = result
            return result
    
    @torch.inference_mode()
    def test_forward_based(self, batch_dict: Dict) -> Dict[str, Any]:
        """
        Test Method 2: forward()-based prediction using REAL batch data.
        
        This directly uses batch data from dataset, mimicking base_strategy.py training logic.
        """
        overwatch.info("=" * 80)
        overwatch.info("TEST 2: forward()-based using real batch data")
        overwatch.info("=" * 80)
        
        try:
            # Move batch to device
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            labels = batch_dict["labels"].to(self.device)
            pixel_values = batch_dict["pixel_values"].to(self.device)
            
            overwatch.debug(f"Input IDs shape: {input_ids.shape}")
            overwatch.debug(f"Labels shape: {labels.shape}")
            overwatch.debug(f"Pixel values shape: {pixel_values.shape}")
            
            # DEBUG: Log input_ids for comparison with predict_action
            overwatch.info(f">> [DATASET DEBUG] Input IDs shape: {input_ids.shape}")
            overwatch.info(f">> [DATASET DEBUG] Input IDs[:20]: {input_ids[0, :20]}")
            overwatch.info(f">> [DATASET DEBUG] Input IDs[-20:]: {input_ids[0, -20:]}")
            
            # Forward pass (no labels for inference)
            autocast_dtype = self.vla.llm_backbone.half_precision_dtype
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vla.enable_mixed_precision_training):
                output = self.vla.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=None,  # Inference mode
                )
            
            overwatch.debug(f"Logits shape: {output.logits.shape}")
            
            # Extract token predictions (EXACTLY matching base_strategy.py line 214-216)
            pred = output.logits[:, self.vla.vision_backbone.num_patches : -1].argmax(dim=2)
            overwatch.debug(f"Pred shape: {pred.shape}")
            
            # Apply mask (EXACTLY matching base_strategy.py line 216)
            # Note: In base_strategy, gt = batch["labels"][:, 1:], then mask = gt > trajectory_token_begin_idx
            # We do the same here
            gt = labels[:, 1:]  # Skip first token (shift for causal LM)
            mask = gt > self.vla.trajectory_converter.trajectory_token_begin_idx
            overwatch.debug(f"Mask shape: {mask.shape}, mask sum: {mask.sum().item()}")
            
            # Extract action tokens (EXACTLY matching base_strategy.py)
            pred_masked = pred[mask].cpu().numpy()
            overwatch.debug(f"Masked pred shape: {pred_masked.shape}")
            overwatch.debug(f"n_dims: {self.vla.trajectory_converter.n_dims}")
            
            # Ensure divisibility by n_dims
            n_dims = self.vla.trajectory_converter.n_dims
            if len(pred_masked) % n_dims != 0:
                overwatch.warning(
                    f"Masked pred length ({len(pred_masked)}) not divisible by n_dims ({n_dims}). "
                    f"Truncating to {(len(pred_masked) // n_dims) * n_dims}"
                )
                pred_masked = pred_masked[: (len(pred_masked) // n_dims) * n_dims]
            
            # Decode to continuous actions
            normalized_actions = self.vla.trajectory_converter.decode_text_ids_to_trajectory(pred_masked)
            overwatch.debug(f"Normalized actions shape: {normalized_actions.shape}")
            
            # Extract first timestep action
            if normalized_actions.ndim == 2:
                action = normalized_actions[0, :]
            elif normalized_actions.ndim == 1:
                action = normalized_actions[:n_dims]
            else:
                raise ValueError(f"Unexpected normalized_actions shape: {normalized_actions.shape}")
            
            result = {
                "status": "success",
                "action_shape": action.shape,
                "action_dtype": str(action.dtype),
                "action_sample": action[:3].tolist(),
                "action_tokens_shape": pred_masked.shape,
                "normalized_actions_shape": normalized_actions.shape,
                "intermediate_states": {
                    "input_ids_shape": input_ids.shape,
                    "labels_shape": labels.shape,
                    "logits_shape": output.logits.shape,
                    "pred_shape": pred.shape,
                    "mask_sum": mask.sum().item(),
                    "pred_masked_shape": pred_masked.shape,
                }
            }
            
            overwatch.info(f"✓ Action shape: {result['action_shape']}")
            overwatch.info(f"✓ Action tokens shape: {result['action_tokens_shape']}")
            
            self.results["forward_based"]["output"] = {
                "action": action,
                "normalized_actions": normalized_actions,
                "action_tokens": pred_masked,
            }
            self.results["forward_based"]["summary"] = result
            
            return result
            
        except Exception as e:
            overwatch.error(f"✗ forward-based method failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "status": "failed",
                "error": str(e),
            }
            self.results["forward_based"]["summary"] = result
            return result
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare the two methods."""
        overwatch.info("=" * 80)
        overwatch.info("COMPARISON RESULTS")
        overwatch.info("=" * 80)
        
        summary_1 = self.results["predict_action"].get("summary", {})
        summary_2 = self.results["forward_based"].get("summary", {})
        
        comparison = {
            "both_successful": (
                summary_1.get("status") == "success" and 
                summary_2.get("status") == "success"
            ),
        }
        
        if comparison["both_successful"]:
            # Compare shapes
            overwatch.info(f"\n[SHAPE COMPARISON]")
            overwatch.info(f"  predict_action action shape: {summary_1['action_shape']}")
            overwatch.info(f"  forward_based action shape: {summary_2['action_shape']}")
            overwatch.info(f"  Shapes match: {summary_1['action_shape'] == summary_2['action_shape']}")
            
            # Compare action values
            action_1 = self.results["predict_action"]["output"]["action"]
            action_2 = self.results["forward_based"]["output"]["action"]
            
            action_diff = np.abs(action_1 - action_2)
            action_l2_norm = np.linalg.norm(action_diff)
            
            comparison["action_l2_distance"] = float(action_l2_norm)
            comparison["action_max_diff"] = float(np.max(action_diff))
            comparison["action_mean_diff"] = float(np.mean(action_diff))
            comparison["actions_close"] = action_l2_norm < 1e-3
            
            overwatch.info(f"\n[VALUE COMPARISON]")
            overwatch.info(f"  Action L2 distance: {comparison['action_l2_distance']:.6e}")
            overwatch.info(f"  Action max difference: {comparison['action_max_diff']:.6e}")
            overwatch.info(f"  Action mean difference: {comparison['action_mean_diff']:.6e}")
            overwatch.info(f"  Actions close (L2 < 1e-3): {comparison['actions_close']}")
            
            # Print sample values
            overwatch.info(f"\n[SAMPLE VALUES]")
            overwatch.info(f"  predict_action action[:3]: {summary_1['action_sample']}")
            overwatch.info(f"  forward_based action[:3]: {summary_2['action_sample']}")
            
            if not comparison["actions_close"]:
                overwatch.warning("\n⚠️  SIGNIFICANT DIFFERENCES DETECTED!")
                diff_indices = np.where(action_diff > 1e-4)[0]
                if len(diff_indices) > 0:
                    overwatch.info(f"  First few indices with diff > 1e-4:")
                    for idx in diff_indices[:5]:
                        overwatch.info(f"    [{idx}] predict={action_1[idx]:.6f}, forward={action_2[idx]:.6f}, diff={action_diff[idx]:.6e}")
            else:
                overwatch.info("\n✓ Actions are virtually identical!")
                
            # Print intermediate state comparison
            overwatch.info(f"\n[INTERMEDIATE STATES]")
            if "intermediate_states" in summary_2:
                inter = summary_2["intermediate_states"]
                overwatch.info(f"  Input IDs shape: {inter['input_ids_shape']}")
                overwatch.info(f"  Labels shape: {inter['labels_shape']}")
                overwatch.info(f"  Logits shape: {inter['logits_shape']}")
                overwatch.info(f"  Pred shape: {inter['pred_shape']}")
                overwatch.info(f"  Mask sum: {inter['mask_sum']}")
                overwatch.info(f"  Masked pred shape: {inter['pred_masked_shape']}")
                
        else:
            if summary_1.get("status") != "success":
                overwatch.error(f"predict_action failed: {summary_1.get('error')}")
            if summary_2.get("status") != "success":
                overwatch.error(f"forward_based failed: {summary_2.get('error')}")
        
        self.results["comparison"] = comparison
        return comparison


def main():
    parser = argparse.ArgumentParser(description="Test predict_action consistency using real dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VLA checkpoint")
    parser.add_argument("--dataset-repo-id", type=str, default="HuggingFaceVLA/libero", help="Dataset repo ID")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0], help="Task IDs to test")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to test")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DatasetActionConsistencyTester(
        checkpoint_path=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        task_ids=args.task_ids,
        device=args.device,
    )
    
    # Get dataset sample
    overwatch.info(f"Loading dataset sample {args.sample_index}...")
    image, instruction, batch_dict = tester.get_dataset_sample(args.sample_index)
    
    # Run both tests
    result_1 = tester.test_predict_action(image, instruction, batch_dict=batch_dict)
    result_2 = tester.test_forward_based(batch_dict)
    
    # Compare results
    comparison = tester.compare_results()
    
    # Exit with appropriate code
    if comparison.get("both_successful") and comparison.get("actions_close"):
        overwatch.info("\n✓ All tests passed! Actions are consistent.")
        return 0
    else:
        overwatch.error("\n✗ Tests failed or actions differ significantly!")
        return 1


if __name__ == "__main__":
    exit(main())
