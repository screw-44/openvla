"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import copy
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import RunStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _deep_copy_to_cpu(obj: Any) -> Any:
    """Recursively deep copy and move tensors to CPU to free GPU memory."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _deep_copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_deep_copy_to_cpu(item) for item in obj)
    else:
        return obj


class DDPStrategy(RunStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.async_save_thread = None
        self.checkpoint_save_error = None

    @staticmethod
    def _move_tensors_to_device(obj, device: str):
        """Recursively move all tensors in a nested dict/list structure to specified device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: DDPStrategy._move_tensors_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(DDPStrategy._move_tensors_to_device(item, device) for item in obj)
        else:
            return obj

    @staticmethod
    def _async_save_worker(
        state_dict_copy: dict,
        optimizer_state_copy: dict,
        checkpoint_path: Path,
        latest_checkpoint_path: Path,
    ) -> None:
        """Background worker thread for saving checkpoints asynchronously."""
        try:
            # Save checkpoint to disk
            torch.save({"model": state_dict_copy, "optimizer": optimizer_state_copy}, checkpoint_path)
            # Copy to latest checkpoint
            shutil.copy(checkpoint_path, latest_checkpoint_path)
        except Exception as e:
            overwatch.error(f"Error during async checkpoint save: {e}")
            raise

    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """
        Save a checkpoint asynchronously to the `run_dir` with GPU memory optimization.
        
        Process:
        Stage 1: Get state_dict from GPU (tensors on GPU)
        Stage 2: Copy + detach to CPU (GPU weights untouched, background save ready)
        Stage 3: Start async disk write in background thread
        """
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        # Wait for any previous async save to complete
        if self.async_save_thread is not None and self.async_save_thread.is_alive():
            overwatch.info(f"Waiting for previous checkpoint save to complete...")
            self.async_save_thread.join()
            if self.checkpoint_save_error is not None:
                raise RuntimeError(f"Previous checkpoint save failed: {self.checkpoint_save_error}")

        # Stage 1: Get state dictionaries from GPU (tensors still on GPU)
        start_time = time.time()
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()
        stage1_time = time.time() - start_time

        # Stage 2: Copy + detach to CPU (GPU weights remain on GPU, ready for next batch)
        copy_to_cpu_start = time.time()
        model_state_dicts_copy = {
            k: v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model_state_dicts.items()
        }
        optimizer_state_copy = {
            k: v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer_state_dict.items()
        }
        stage2_time = time.time() - copy_to_cpu_start

        # Set Checkpoint Path
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        latest_checkpoint_path = checkpoint_dir / "latest-checkpoint.pt"

        # Stage 3: Start async save thread (disk I/O happens in background)
        self.async_save_thread = threading.Thread(
            target=self._async_save_worker,
            args=(model_state_dicts_copy, optimizer_state_copy, checkpoint_path, latest_checkpoint_path),
            daemon=False,
        )
        self.async_save_thread.start()

        overwatch.info(
            f"✅ Checkpoint save started (async) at step {global_step}\n"
            f"         Stage 1 [Get state_dict]:         {stage1_time:.3f}s\n"
            f"         Stage 2 [Copy+detach→CPU]:        {stage2_time:.3f}s (GPU weights untouched)\n"
            f"         Stage 3 [Async disk write]:       In background...\n"
            f"         |-> Total setup time: {stage1_time + stage2_time:.3f}s\n"
            f"         |-> Checkpoint: {checkpoint_path.name}"
        )


    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()

        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)
