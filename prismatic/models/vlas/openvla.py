"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.conf.datasets import DatasetConfig
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.tokenizer import BaseTrajectoryConverter

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        trajectory_converter: BaseTrajectoryConverter,
        dataset_id: str = "libero",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trajectory_converter = trajectory_converter
        self.dataset_id = dataset_id

    @torch.inference_mode()
    def predict_action(
         self, image: Image, instruction: str, **kwargs: str
    ) -> Dict[str, np.ndarray]:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).
        
        [FORWARD-BASED APPROACH] Uses forward() directly instead of generate() for consistent training/inference logic.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string

        @return Dictionary containing:
                - 'action_tokens': predicted action token IDs (sampled from logits)
                - 'action': normalized action vector in [-1, 1] range (from trajectory_converter)
                - 'normalized_actions': same as action
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_builder.add_turn(role="gpt", message="")  # ← 添加这一行来匹配训练结构
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs - get both input_ids and attention_mask
        tokenized = tokenizer(prompt_text, truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                # Also extend attention_mask
                attention_mask = torch.cat(
                    (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # === Single Forward Pass to Get Action Token Logits ===
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,  # Inference mode - no labels needed
            )

        # Extract all token IDs from predictions (matching training code)
        # output.logits shape: [batch=1, seq_len, vocab_size]
        # Training does: output.logits[:, num_patches:-1].argmax(dim=2)

        pred = output.logits[:, self.vision_backbone.num_patches : -1].argmax(dim=2)
        
        # Apply mask to filter action tokens (matching training code)
        mask = pred > self.trajectory_converter.trajectory_token_begin_idx
        trajectory_token_begin_idx = self.trajectory_converter.trajectory_token_begin_idx
        
        # Extract only the masked tokens
        action_token_ids = pred[mask].cpu().numpy()
        
        if len(action_token_ids) == 0:
            overwatch.warn("No action tokens found in predictions (all tokens below trajectory_token_begin_idx)")
            raise ValueError(
                f"No valid action tokens found. trajectory_token_begin_idx={trajectory_token_begin_idx}, "
                f"pred range: [{pred.min()}, {pred.max()}]"
            )
        
        overwatch.info(f"Extracted {len(action_token_ids)} action tokens from {pred.numel()} total token positions")
        
        # If not divisible by n_dims, truncate to nearest multiple
        n_dims = self.trajectory_converter.n_dims
        if len(action_token_ids) % n_dims != 0:
            overwatch.warning(f"Action tokens length ({len(action_token_ids)}) not divisible by n_dims ({n_dims})")
            truncated_len = (len(action_token_ids) // n_dims) * n_dims
            overwatch.warning(f"Truncating from {len(action_token_ids)} to {truncated_len} tokens")
            action_token_ids = action_token_ids[:truncated_len]
        
        # Decode action tokens to continuous action vector
        # This will return shape [n_sequence_length, n_dims]
        try:
            normalized_actions = self.trajectory_converter.decode_text_ids_to_trajectory(action_token_ids)
        except Exception as e:
            overwatch.error(
                f"Failed to decode action tokens (shape={action_token_ids.shape}): {e}"
            )
            raise

        # Extract action for the first timestep (single step prediction)
        # normalized_actions shape is typically [n_sequence_length, n_dims]
        # We take the first timestep's n_dims values as the action
        n_dims = self.trajectory_converter.n_dims
        if normalized_actions.ndim == 2:
            # [n_sequence_length, n_dims] - take first row
            action = normalized_actions[0, :]
        elif normalized_actions.ndim == 1:
            # [n_sequence_length * n_dims] - take first n_dims values
            action = normalized_actions[:n_dims]
        else:
            raise ValueError(f"Unexpected normalized_actions shape: {normalized_actions.shape}")

        overwatch.debug(
            f"Action tokens shape: {action_token_ids.shape}, "
            f"Normalized actions shape: {normalized_actions.shape}, "
            f"Final action shape: {action.shape}"
        )

        return {
            'action_tokens': action_token_ids,
            'action': action,
            'normalized_actions': normalized_actions,
        }

