"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast, GenerationMixin

from core.models.vlms.prismatic import PrismaticVLM
from core.util.overwatch import initialize_overwatch
from core.vla.tokenizer import BaseTrajectoryConverter

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class VLA(PrismaticVLM):
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
    def generate(
        self,
        image: Image.Image,
        prompt_text: str,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate action tokens from image and prompt using parent class's generate_ids interface.
        Supports variable-length output with EOS token.
        """
        tokenizer = self.llm_backbone.tokenizer

        # CRITICAL FIX: The training data includes a trailing space token (29871) after "ASSISTANT:"
        # The prompt_builder.get_prompt() calls .rstrip() which removes this space.
        # We need to add it back to match the exact training format.
        # Training format: "... USER: <instruction>? ASSISTANT: " (with space, token 29871)
        # Without this space, the model generates from the wrong position.
        overwatch.debug(f"Before fix, prompt ends with: {repr(prompt_text[-30:])}")
        if not prompt_text.endswith(" "):
            prompt_text = prompt_text + " "
            overwatch.debug(f"Added space, now ends with: {repr(prompt_text[-30:])}")
        else:
            overwatch.debug("Already has space")

        # Get prompt length for later extraction of generated tokens
        prompt_input_ids = tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids
        prompt_len = prompt_input_ids.shape[1]
        overwatch.debug(
            f"Prompt length = {prompt_len}, last 3 tokens: {prompt_input_ids[0, -3:].tolist()}"
        )

        # NOTE: 12.17 Add eos_token_id for variable-length trajectory generation
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = tokenizer.eos_token_id

        # Use parent class's generate_ids interface
        generated_ids = super().generate_ids(image, prompt_text, **kwargs)

        # Strip prompt to get newly generated tokens
        generated_token_ids_np = (
            generated_ids[0, prompt_len:].cpu().numpy().astype(np.int64)
        )

        # Decode to continuous actions (handles EOS truncation internally)
        normalized_actions = self.trajectory_converter.decode_text_ids_to_trajectory(
            generated_token_ids_np
        )

        # Extract first timestep
        action = (
            normalized_actions[0, :]
            if normalized_actions.ndim == 2
            else normalized_actions[: self.trajectory_converter.n_dims]
        )

        return {
            "action_tokens": generated_token_ids_np,
            "action": action,
            "normalized_actions": normalized_actions,
        }
