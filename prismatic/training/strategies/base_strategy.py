"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.util.overwatch import initialize_overwatch
from prismatic.training.metrics import VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction,
    PaddedCollatorForLanguageModeling,
)


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

DATA_LOADER_NUM_WORKERS = 12


# === Abstract Base Class for an arbitrary Training Strategy ===
class RunStrategy(ABC):
    def __init__(
        self,
        vla: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.vla, self.device_id, self.stage = vla, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = (
            self.vla.all_module_keys,
            self.vla.trainable_module_keys,
        )
        self.llm_transformer_layer_cls = self.vla.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = (
            global_batch_size,
            per_device_batch_size,
        )

        self.learning_rate, self.weight_decay, self.max_grad_norm = (
            learning_rate,
            weight_decay,
            max_grad_norm,
        )
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"

        self.grad_accumulation_steps = (
            self.global_batch_size
            // self.per_device_batch_size
            // overwatch.world_size()
        )
        if self.enable_mixed_precision_training:
            assert (
                self.mixed_precision_dtype == torch.bfloat16
            ), "Only BF16 mixed precision training is supported!"
            assert (
                check_bloat16_supported()
            ), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    # === VLA Training ===

    def train_vla(
        self,
        vla_dataset: Dataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        run_dir: Path,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """
        运行VLA训练循环

        Args:
            vla_dataset: 训练数据集
            collator: 批处理数据整理器
            metrics: 指标追踪器
            run_dir: 检查点保存目录
            save_interval: 检查点保存间隔（步数）
            save_full_model: 是否保存完整模型（否则只保存可训练参数）
            mode_config: 可选的验证模式配置
            max_steps: 最大训练步数（设置后将覆盖epochs限制）
        """
        # 创建数据加载器（使用多worker并行加载提升效率）
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=DATA_LOADER_NUM_WORKERS,
            worker_init_fn=self.worker_init_fn,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

        # === 开始训练 ===
        self.vla.train()
        accumulated_steps = 0
        for epoch in range(self.epochs):
            # 内层进度条显示当前轮的训练进度
            step_progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=not overwatch.is_rank_zero(),
                initial=metrics.global_step % len(dataloader),  # resume后的起始
            )
            for batch in step_progress:
                metrics.global_step += 1
                with torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision_training,
                ):
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                
                # 反向传播（梯度累积）
                loss.backward()
                accumulated_steps += 1

                # === 计算动作预测准确率和L1损失 ===
                if overwatch.is_rank_zero():
                    with torch.no_grad():
                        metrics.log_pro(
                            output, batch, self.vla, self.lr_scheduler.get_last_lr()[0]
                        )

                # === 梯度累积：达到累积步数或最后一个batch时执行optimizer.step ===
                if accumulated_steps >= self.grad_accumulation_steps:
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    accumulated_steps = 0

                step_progress.set_description(
                    f"[Epoch {epoch}] G_Step {metrics.global_step} | Lr:{self.lr_scheduler.get_last_lr()[0]:.6f} | Loss:{loss.item():.4f}"
                )
                
                # 定期保存检查点
                if metrics.global_step % save_interval == 0:
                    overwatch.info(f"保存checkpoint在{metrics.global_step}")
                    self.save_checkpoint(
                        run_dir,
                        metrics.global_step,
                        epoch,
                        loss.item(),
                        only_trainable=not save_full_model,
                    )
                    dist.barrier()

                if (
                    self.max_steps is not None and metrics.global_step >= self.max_steps
                ):
                    # 最后一次更新（处理剩余梯度）
                    if accumulated_steps > 0:
                        self.clip_grad_norm()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    overwatch.info(f"max steps已经达到，完成训练。 保存最终检查点中")
                    self.save_checkpoint(
                        run_dir,
                        metrics.global_step,
                        epoch,
                        loss.item(),
                        only_trainable=not save_full_model,
                    )
                    dist.barrier()
                    return

            # Epoch 结束时处理剩余梯度
            if accumulated_steps > 0:
                self.clip_grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad()

        overwatch.info(f"完成全部 {self.epochs} 轮训练。保存最终检查点...")
        self.save_checkpoint(
            run_dir,
            metrics.global_step,
            epoch,
            loss.item(),
            only_trainable=not save_full_model,
        )
        dist.barrier()
        return

    def _compute_sample_l1_errors(self, predictions, ground_truths):
        """
        计算样本的L1误差列表（处理变长轨迹）

        Args:
            predictions: 预测轨迹列表
            ground_truths: 真实轨迹列表

        Returns:
            L1误差列表
        """
        l1_errors = []
        for i in range(len(predictions)):
            try:
                pred_array = np.array(predictions[i])
                gt_array = np.array(ground_truths[i])

                # 确保至少是1维
                if pred_array.ndim == 0:
                    pred_array = np.array([pred_array])
                if gt_array.ndim == 0:
                    gt_array = np.array([gt_array])

                # 获取长度
                pred_len = (
                    pred_array.shape[0] if pred_array.ndim > 1 else len(pred_array)
                )
                gt_len = gt_array.shape[0] if gt_array.ndim > 1 else len(gt_array)

                # 取最小长度进行比较（因为EOS可能导致长度不同）
                min_len = min(pred_len, gt_len)
                if min_len > 0:
                    l1_error = float(
                        np.abs(pred_array[:min_len] - gt_array[:min_len]).mean()
                    )
                else:
                    l1_error = 0.0
                l1_errors.append(l1_error)
            except Exception as e:
                overwatch.warning(f"计算样本 {i} 的L1误差失败: {e}")
                l1_errors.append(0.0)

        return l1_errors
