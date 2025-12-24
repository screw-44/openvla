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
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction,
    PaddedCollatorForLanguageModeling,
)
from prismatic.conf import ModeConfig


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
        mode_config: Optional[ModeConfig] = None,
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
        assert self.grad_accumulation_steps == 1, "VLA训练不支持梯度累积！"

        # 设置数据集为训练模式
        vla_dataset.get_train_data = True

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
                # batch["input_ids"][0][-8] = 100
                # overwatch.info(
                #     f"training batch: input_ids[0]={batch['input_ids'][0]}, shape={batch['input_ids'][0].shape}, "
                #     f"attention_mask[0]={batch['attention_mask'][0]}, shape {batch['attention_mask'][0].shape}"
                #     f"labels[0]={batch['labels'][0]},  shape {batch['labels'][0].shape}"
                #     # f"cam1={batch['pixel_values']['cam1'][0]} "
                #     # f"cam2[0]={batch['pixel_values']['cam2'][0]}"
                # )
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
                loss.backward()  # 反向传播
                # overwatch.info(f"output logits:{output.logits.argmax(dim=2)[0][-10:]}")
                # print("input prompt:", self.model.llm_backbone.tokenizer.decode(input_ids.squeeze(0).tolist()))
                # generated_ids = generated_ids[0, input_ids.shape[1] :].cpu()

                # === 计算动作预测准确率和L1损失 ===
                # 使用metrics.log_pro()计算所有指标（处理变长序列和per-sample L1损失）
                if overwatch.is_rank_zero():
                    with torch.no_grad():
                        metrics.log_pro(
                            output, batch, self.vla, self.lr_scheduler.get_last_lr()[0]
                        )

                # === 梯度更新 ===
                self.clip_grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

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

                # 定期运行验证（在global_step更新后检查）
                # if mode_config is not None and mode_config.has_validate:
                #     if metrics.global_step % mode_config.validate_interval == 0:
                #         overwatch.info(f"在步数 {metrics.global_step} 处运行验证...")
                #         self.validate_vla(vla_dataset, collator, metrics, mode_config=mode_config)
                #          # 验证后恢复训练模式
                #         self.vla.train()
                #         overwatch.info("验证完成，恢复训练模式。")

                if (
                    self.max_steps is not None and metrics.global_step >= self.max_steps
                ):  # 如果这个参数没有设置，就不用比较
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

    # === VLA validate ===

    def validate_vla(
        self,
        vla_dataset: Dataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: "VLAMetrics",
        mode_config: ModeConfig,
    ) -> None:
        """
        运行VLA验证循环，记录损失和动作指标到trackio
        """
        # 设置数据集为验证模式
        vla_dataset.get_train_data = False

        # 初始化数据收集容器
        all_losses = []
        all_action_accuracies = []
        sample_predictions = []
        sample_gts = []
        max_samples = 4

        # 创建验证数据加载器
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

        # === 验证循环 ===
        self.vla.eval()

        total_batches_processed = 0
        max_batches = min(
            mode_config.validate_data_length,
            len(vla_dataset) // self.per_device_batch_size,
        )

        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"验证中",
            total=max_batches,
            disable=not overwatch.is_rank_zero(),
        )

        for batch_idx, batch in progress_bar:
            if total_batches_processed >= mode_config.validate_data_length:
                break

            total_batches_processed += 1

            # 前向传播（验证时不需要梯度）
            with (
                torch.no_grad(),
                torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision_training,
                ),
            ):
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )

            metric_dict = metrics.log_pro(output, batch, self.vla, global_step=0)
            action_accuracy = metric_dict["action_accuracy"]

            # 收集指标（仅在rank 0）
            if overwatch.is_rank_zero():
                all_losses.append(output.loss.item())
                all_action_accuracies.append(action_accuracy.item())

                # 解码并收集样本（用于可视化）
                # 重新计算mask和preds用于样本收集
                IGNORE_INDEX = -100
                action_mask = batch["labels"] != IGNORE_INDEX
                starts = action_mask.float().argmax(dim=1)
                min_start = int(starts.min().item()) if action_mask.any() else 0

                if batch["labels"].size(1) > min_start:
                    action_logits = output.logits[:, min_start:, :]
                    action_preds = action_logits.argmax(dim=2)
                    gt = batch["labels"][:, min_start:]

                    # 对齐长度
                    if action_preds.size(1) != gt.size(1):
                        min_len = min(action_preds.size(1), gt.size(1))
                        action_preds = action_preds[:, :min_len]
                        gt = gt[:, :min_len]

                    mask = gt > self.vla.trajectory_converter.trajectory_token_begin_idx

                    # 收集样本
                    batch_size = action_preds.shape[0]
                    for sample_idx in range(batch_size):
                        if len(sample_predictions) >= max_samples:
                            break

                        sample_pred_ids = (
                            action_preds[sample_idx][mask[sample_idx]].cpu().numpy()
                        )
                        sample_gt_ids = gt[sample_idx][mask[sample_idx]].cpu().numpy()

                        try:
                            continuous_pred = self.vla.trajectory_converter.decode_text_ids_to_trajectory(
                                sample_pred_ids
                            )
                            continuous_gt = self.vla.trajectory_converter.decode_text_ids_to_trajectory(
                                sample_gt_ids
                            )
                            sample_predictions.append(continuous_pred.tolist())
                            sample_gts.append(continuous_gt.tolist())
                        except Exception as e:
                            overwatch.warning(
                                f"批次{batch_idx}样本{sample_idx}解码失败: {e}"
                            )

        # === 记录验证指标到trackio（仅在rank 0） ===
        if overwatch.is_rank_zero() and len(all_losses) > 0:
            # 计算平均指标
            avg_loss = torch.tensor(all_losses).mean().item()
            avg_accuracy = torch.tensor(all_action_accuracies).mean().item()

            # 计算每个样本的L1误差（需要处理变长轨迹）
            l1_errors = self._compute_sample_l1_errors(sample_predictions, sample_gts)

            # 记录验证样本和表格到trackio
            metrics.log_validation_samples(
                predictions=sample_predictions,
                ground_truths=sample_gts,
                max_samples=len(sample_predictions),
                dataset_name=None,
            )

            metrics.log_validation_table(
                avg_loss=avg_loss,
                avg_accuracy=avg_accuracy,
                l1_errors=l1_errors,
                dataset_name=None,
            )

            overwatch.info(f"✅ 验证指标已记录到trackio（步数 {metrics.global_step}）")
            overwatch.info(f"   - 损失: {avg_loss:.4f}")
            overwatch.info(f"   - 动作准确率: {avg_accuracy:.4f}")
            overwatch.info(f"   - 记录了 {len(sample_predictions)} 个预测样本")

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
