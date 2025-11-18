"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging.
Now uses trackio (imported as wandb) for experiment tracking.
"""
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import jsonlines
import numpy as np
import torch
import trackio as wandb

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class Tracker:
    """Tracker using trackio (imported as wandb) for experiment tracking"""
    def __init__(
        self,
        run_id: str,
        hparams: Dict[str, Any],
        group: str = "align",  # 项目里面的一类实验
        project: str = "vla-training", # 一个大项目的名称
    ) -> None:
        self.run_id, self.hparams = run_id, hparams

        # Get trackio initialization parameters
        self.group = group
        self.project = project

        # Call trackio.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        # trackio.init() requires 'project' as a positional argument
        wandb.init(
            project=self.project,
            name=self.run_id, # name 就是 run_id(需要设立完全可读的run_id)
            config=self.hparams,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)


# === Core Metrics Container :: Initializes Trackers => Compiles/Pushes Metrics ===

class VLAMetrics:
    def __init__(
        self,
        run_id: str,
        hparams: Dict[str, Any],
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
        project: str = "vla-training",
    ) -> None:
        self.run_id, self.hparams = run_id, hparams

        # Initialize Trackers
        self.tracker = Tracker(run_id, hparams, group="vla-train", project=project)
        self.tracker.write_hyperparameters()

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "l1_loss": deque(maxlen=window_size),
            "action_accuracy": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

        # Created metrics buffers for individual tracked datasets
        # 注意：这里不应该创建完整的 VLAMetrics 对象，只需要创建简单的 state 字典
        def create_dataset_tracker():
            return {
                "l1_loss": deque(maxlen=window_size),
                "action_accuracy": deque(maxlen=window_size),
            }
        self.dataset_trackers = defaultdict(create_dataset_tracker)

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        self.tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} - Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    def commit_for_dataset(self, dataset_name: str, **kwargs) -> None:
        # 直接将数据添加到对应数据集的 tracker 中
        tracker_state = self.dataset_trackers[dataset_name]
        for key, value in kwargs.items():
            if key in tracker_state:
                tracker_state[key].append(value.detach() if hasattr(value, 'detach') else value)

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = float(torch.stack(list(self.state["loss_raw"])).mean().item())
        loss = float(torch.stack(list(self.state["loss"])).mean().item())
        l1_loss = float(torch.stack(list(self.state["l1_loss"])).mean().item())
        action_accuracy = float(torch.stack(list(self.state["action_accuracy"])).mean().item())
        step_time = float(np.mean(list(self.state["step_time"])))
        lr = float(self.state["lr"][-1])
        status = self.get_status(loss)

        # Get metrics per dataset
        dataset_metrics = {}
        for ds, tracker_state in self.dataset_trackers.items():
            # 检查是否有数据
            if len(tracker_state["l1_loss"]) > 0 and len(tracker_state["action_accuracy"]) > 0:
                dataset_metrics.update(
                    {
                        f"{ds}/L1 Loss": float(torch.stack(list(tracker_state["l1_loss"])).mean().item()),
                        f"{ds}/Action Token Accuracy": float(torch.stack(list(tracker_state["action_accuracy"])).mean().item()),
                    }
                )

        # Fire to Trackers
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Loss": loss,
                f"{prefix}/L1 Loss": l1_loss,
                f"{prefix}/Action Token Accuracy": action_accuracy,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
                **dataset_metrics,
            },
        )
        return status

    def finalize(self) -> str:
        self.tracker.finalize()











# 我们train.py并不涉及到的部分
class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        stage: str,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        self.tracker = Tracker(run_id, hparams, group=self.stage, project="vla-training")
        self.tracker.write_hyperparameters()

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        self.tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self, *, global_step: Optional[int] = None, lr: Optional[float] = None, update_step_time: bool = False, **kwargs
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        step_time, lr = np.mean(list(self.state["step_time"])), self.state["lr"][-1]
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
            },
        )
        return status

    def finalize(self) -> str:
        self.tracker.finalize()

