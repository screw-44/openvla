"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

import trackio
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.tokenizer import BaseTrajectoryConverter
from prismatic.conf import ModeConfig


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

DATA_LOADER_NUM_WORKERS = 8

# === Abstract Base Class for an arbitrary Training Strategy ===
class RunStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
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
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
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

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

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
        collator: PaddedCollatorForActionPrediction, # TODO 看一下是否兼容
        trajectory_converter: BaseTrajectoryConverter,
        metrics: VLAMetrics,
        run_dir: Path,
        save_interval: int = 2500,
        save_full_model: bool = True,
        mode_config: Optional[ModeConfig] = None,
        epochs: int = 10,
        max_steps: Optional[int] = None,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`.
        
        Args:
            vla_dataset: Training dataset
            collator: Data collator for batching
            trajectory_converter: Converter for trajectory encoding/decoding
            metrics: Metrics tracker
            run_dir: Directory path for saving checkpoints
            save_interval: Interval (in steps) for saving checkpoints
            save_full_model: Whether to save the full model or just trainable params
            mode_config: Optional mode configuration for validation during training
            epochs: Maximum number of epochs to train (default: 1000)
            max_steps: Maximum number of gradient steps to train (overrides epochs if set)
        """
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"
        
        # Determine training limits
        # If max_steps is set, use it; otherwise use epochs
        use_max_steps = max_steps is not None
        target_epochs = epochs if not use_max_steps else float('inf')  # Infinite epochs if using max_steps
        target_max_steps = max_steps if use_max_steps else None
        
        overwatch.info(f"Training Configuration:")
        if use_max_steps:
            overwatch.info(f"  - Max Steps: {target_max_steps} (training will stop when reaching this step count)")
        else:
            overwatch.info(f"  - Epochs: {target_epochs} (training will stop after {target_epochs} epochs)")
        
        # 设置拿到train的数据
        vla_dataset.get_train_data = True
        # Create a DataLoader with multi-worker data loading for efficiency
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=DATA_LOADER_NUM_WORKERS,  # 简单设置，后续可以调优
            worker_init_fn=self.worker_init_fn,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

        # === Train ===
        status = metrics.get_status()
        
        # Use parameterized epochs instead of self.epochs
        max_epochs = target_epochs if target_epochs != float('inf') else 999999  # Large number for display
        epoch_progress = tqdm(
            range(int(max_epochs)), desc=f"Epochs", leave=True, disable=not overwatch.is_rank_zero(), position=0
        )
        for epoch in epoch_progress:
            self.vlm.train()
            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # Inner progress bar for steps within each epoch
            step_progress = tqdm(
                enumerate(dataloader), desc=f"Epoch {epoch+1}/{max_epochs} - {status}",
                leave=False, disable=not overwatch.is_rank_zero(), total=len(dataloader), position=1
            )

            for train_idx, batch in step_progress:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(pred == logits) & mask` --> sum/divide by # unmasked!
                pred = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                gt = batch["labels"][:, 1:].to(pred.device)
                mask = gt > trajectory_converter.trajectory_token_begin_idx

                # Compute Accuracy
                pred = (pred == gt) & mask
                action_accuracy = pred.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_pred = torch.tensor(
                    trajectory_converter.decode_text_ids_to_trajectory(pred[mask].cpu().numpy())
                )
                continuous_gt = torch.tensor(
                    trajectory_converter.decode_text_ids_to_trajectory(gt[mask].cpu().numpy())
                )
                # print(f"continous_actions_pred - continous_actions_gt [0:5]: {continuous_pred[0:5] - continuous_gt[0:5]}")

                action_l1_loss = torch.nn.functional.l1_loss(continuous_pred, continuous_gt)
                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Save model based on cfg.save_interval
                if (metrics.global_step + 1) % save_interval == 0:
                    self.save_checkpoint(
                        run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                # Do Validation if specified
                if mode_config is not None and mode_config.is_validate:
                    if metrics.global_step % mode_config.validate_interval == 0:
                        overwatch.info(f"Running Validation at Step {metrics.global_step}...")
                        self.validate_vla(
                            vla_dataset, # TODO： 添加训练数据集/测试数据集/验证数据集的区分
                            collator,
                            trajectory_converter,
                            metrics,
                            mode_config=mode_config,
                        )
                        overwatch.info("Done with Validation.")

                # Update step Progress Bars
                step_progress.set_description(f"Epoch {epoch+1}/{max_epochs} - {status}")
                
                # Check if we've reached max_steps limit
                if target_max_steps is not None and metrics.global_step >= target_max_steps:
                    overwatch.info(f"Reached max_steps limit ({target_max_steps}). Saving final checkpoint and exiting...")
                    
                    # Save final checkpoint before exit
                    self.save_checkpoint(
                        run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()
                    
                    overwatch.info(f"Training completed at step {metrics.global_step}")
                    return  # Exit the training function

            # (Epoch级别暂时没有记录) Update epoch progress bar after finishing all steps in this epoch
            epoch_progress.set_description(f"Epochs - {status}")
            
            # Check if we've reached epochs limit (if not using max_steps mode)
            if not use_max_steps and (epoch + 1) >= target_epochs:
                overwatch.info(f"Completed all {target_epochs} epochs. Saving final checkpoint...")
                
                # Save final checkpoint
                self.save_checkpoint(
                    run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                )
                dist.barrier()
                
                overwatch.info(f"Training completed after {target_epochs} epochs")
                return  # Exit the training function

    # === VLA Testing ===

    def validate_vla(
        self,
        vla_dataset: Dataset,
        collator: PaddedCollatorForActionPrediction,
        trajectory_converter: BaseTrajectoryConverter,
        metrics: "VLAMetrics",
        mode_config: ModeConfig,
    ) -> None:
        """Run the VLA validation loop for the given `dataset` and `collator`; log losses, action metrics to trackio."""
        # 设置拿到test的数据
        vla_dataset.get_train_data = False

        # Initialize data collection for trackio logging
        all_losses = []
        all_action_accuracies = []
        sample_predictions = []
        sample_gts = []
        max_samples = 4  # Only collect 4 samples for the table

        # 注意shuffle是开的
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

        # === Testing ===
        self.vlm.eval()

        
        # Validation loop: iterate over epochs and batches
        total_batches_processed = 0
        for epoch in range(1):  # 只验证第一个epoch
            epoch_progress = tqdm(
                enumerate(dataloader), 
                desc=f"Validation Epoch {epoch+1}",
                total=min(mode_config.validate_data_length, len(vla_dataset) // self.per_device_batch_size),
                disable=not overwatch.is_rank_zero()
            )
            
            for batch_idx, batch in epoch_progress:
                # 手动控制测试长度
                if total_batches_processed >= mode_config.validate_data_length:
                    break
                
                total_batches_processed += 1
                
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )

                # === Compute Action Token Accuracy & L1 Loss ===
                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(pred == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = gt > trajectory_converter.trajectory_token_begin_idx

                # Compute Accuracy
                pred = (action_preds == gt) & mask
                action_accuracy = pred.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_pred = torch.tensor(
                    trajectory_converter.decode_text_ids_to_trajectory(action_preds[mask].cpu().numpy())
                )
                continuous_gt = torch.tensor(
                    trajectory_converter.decode_text_ids_to_trajectory(gt[mask].cpu().numpy())
                )
                
                # Collect metrics for trackio logging (only on rank 0)
                if overwatch.is_rank_zero():
                    all_losses.append(output.loss.item())
                    all_action_accuracies.append(action_accuracy.item())
                    
                    # Collect sample predictions and ground truth (only first 4)
                    if len(sample_predictions) < max_samples:
                        sample_predictions.append(continuous_pred.cpu().numpy().tolist())
                        sample_gts.append(continuous_gt.cpu().numpy().tolist())

        # === Log validation metrics to trackio (only on rank 0) ===
        if overwatch.is_rank_zero() and len(all_losses) > 0:
            # Compute average metrics 注意要转换到python类型
            avg_loss = torch.tensor(all_losses).mean().item()
            avg_accuracy = torch.tensor(all_action_accuracies).mean().item()
            
            # Create prediction samples table
            table_data = []
            for i in range(len(sample_predictions)):
                pred_str = str(sample_predictions[i][:5]) + '...' if len(sample_predictions[i]) > 5 else str(sample_predictions[i])
                gt_str = str(sample_gts[i][:5]) + '...' if len(sample_gts[i]) > 5 else str(sample_gts[i])
                l1_error = float(np.abs(np.array(sample_predictions[i]) - np.array(sample_gts[i])).mean())
                table_data.append([i, pred_str, gt_str, f"{l1_error:.4f}"])
            
            prediction_table = trackio.Table(
                columns=['Sample ID', 'Prediction', 'Ground Truth', 'L1 Error'],
                data=table_data
            )
            
            # Build validation metrics dict
            validation_metrics = {
                'VLA Validation/loss': avg_loss,
                'VLA Validation/action_accuracy': avg_accuracy,
                'VLA Validation/prediction_samples': prediction_table,
            }
            
            # Log to trackio using VLAMetrics
            metrics.log(
                global_step=metrics.global_step,
                metrics=validation_metrics
            )
            
            overwatch.info(f"✅ Validation metrics logged to trackio at step {metrics.global_step}")
            overwatch.info(f"   - Loss: {avg_loss:.4f}")
            overwatch.info(f"   - Action accuracy: {avg_accuracy:.4f}")
            overwatch.info(f"   - Logged {len(sample_predictions)} prediction samples")

                #     # print(visual_img.min().item(), " ", visual_img.max().item())
                #     # 调整通道顺序 (C, H, W) → (H, W, C)
                #     visual_img = np.transpose(visual_img, (1, 2, 0))      # (224, 224, 3) (H, W, 3) 
                #     visual_img = (visual_img + 1) / 2
                #     visual_img = (visual_img * 255).astype(np.uint8)
                #     # print(visual_img.min().item(), " ", visual_img.max().item())
                #     visual_img = cv2.resize(visual_img, (1280, 720))
                #     # 读取相机内参 ob camera
                #     camera_K = np.array(
                #         [[685.95849609375, 0.0, 644.1708984375],
                #         [0.0, 686.1210327148438, 362.3411560058594],
                #         [0.0, 0.0, 1.0]]
                #     )
                #     trajectory_pred = SE3Converter.d9_to_se3(save_data_batch["aff_pred"])
                #     pose_6d_pred = SE3Converter.d9_to_se3(save_data_batch["object_pose_pred"])
                #     trajectory_gt = SE3Converter.d9_to_se3(save_data_batch["aff_gt"])
                #     # print("trajectory_gt shape:", trajectory_gt.shape)
                #     pose_6d_gt = SE3Converter.d9_to_se3(save_data_batch["object_pose_gt"])
                #     # print("object pose shape:", pose_6d_gt.shape)

                #     # 如果是object centircs,这里trajectory的key是object
                #     trajectory_key = "object"
                #     visualized_img_pred = draw_aff_on_image(
                #         image=visual_img,
                #         object_pose=pose_6d_pred,
                #         trajectory=trajectory_pred,
                #         K=camera_K,
                #         object_gt_pose=pose_6d_gt, # 使用gt的物体位姿作为参考
                #     )

                #     visualized_img_gt = draw_aff_on_image(
                #         image=visual_img,
                #         object_pose=pose_6d_gt,
                #         trajectory=trajectory_gt,
                #         K=camera_K,
                #         object_gt_pose=pose_6d_gt, # 使用gt的物体位姿作为参考
                #     )

                #     # === 左右拼接图像并添加文字标签 ===
                #     # 左右拼接图像
                #     combined_img = np.hstack([visualized_img_pred, visualized_img_gt])
                    
                #     # 在图像顶部添加文字标签
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     font_scale = 1.5
                #     font_color = (255, 255, 255)  # 白色
                #     font_thickness = 3
                    
                #     # 计算文字位置
                #     text_pred = "PRED"
                #     text_gt = "GT"
                #     text_size_pred = cv2.getTextSize(text_pred, font, font_scale, font_thickness)[0]
                #     text_size_gt = cv2.getTextSize(text_gt, font, font_scale, font_thickness)[0]
                    
                #     # 在左半部分（pred）顶部居中添加"PRED"
                #     x_pred = (1280 - text_size_pred[0]) // 2
                #     y_pred = 40  # 距离顶部40像素
                #     cv2.putText(combined_img, text_pred, (x_pred, y_pred), font, font_scale, font_color, font_thickness)
                    
                #     # 在右半部分（gt）顶部居中添加"GT"
                #     x_gt = 1280 + (1280 - text_size_gt[0]) // 2
                #     y_gt = 40  # 距离顶部40像素
                #     cv2.putText(combined_img, text_gt, (x_gt, y_gt), font, font_scale, font_color, font_thickness)

                #     # # 保存原始单独图像
                #     # cv2.imwrite(str(vis_save_dir / f"test_{total_batches_processed-1:05d}_pred.png"), visualized_img_pred)
                #     # cv2.imwrite(str(vis_save_dir / f"test_{total_batches_processed-1:05d}_gt.png"), visualized_img_gt)
                    
                #     combined_img = cv2.resize(combined_img, None, fx=0.3, fy=0.3)  #压缩图像大小到接近到224的地方
                #     # 保存拼接后的图像
                #     cv2.imwrite(str(vis_save_dir / f"test_{total_batches_processed-1:05d}_combined.jpg"), combined_img)




# Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
# 多个数据集分别看收敛的曲线，还是比较实用的，保留了这个dataset_names的属性。
# 这里实现遇到问题，看上去是在PaddedCollatorForActionPrediction里面没有添加这个属性进行传递，目前暂时关闭了这个功能
# if overwatch.is_rank_zero():
#     datasets = set(batch["dataset_names"])
#     if len(datasets) > 1:
#         for ds in datasets:
#             ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
#             action_accuracy_ds = pred[ds_mask].sum().float() / mask[ds_mask].sum().float()
#             continuous_actions_pred_ds = torch.tensor(
#                 trajectory_converter.decode_text_ids_to_trajectory(
#                     action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
#                 )
#             )
#             continuous_actions_gt_ds = torch.tensor(
#                 trajectory_converter.decode_text_ids_to_trajectory(
#                     gt[ds_mask][mask[ds_mask]].cpu().numpy()
#                 )
#             )
#             action_l1_loss_ds = torch.nn.functional.l1_loss(
#                 continuous_actions_pred_ds, continuous_actions_gt_ds
#             )
#             metrics.commit_for_dataset(
#                 dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
#             )




    # def run_training(
    #     self,
    #     dataset: Dataset,
    #     collator: PaddedCollatorForLanguageModeling,
    #     metrics: Metrics,
    #     stage: str = "finetune",
    #     batch_construction_strategy: str = "split-modality",
    #     seed: int = 7,
    # ) -> None:
    #     """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
    #     if "finetune" in stage and batch_construction_strategy == "split-modality":
    #         # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
    #         #   (e.g., grouping by length) =>> can easily add them here!
    #         modality_lengths = dataset.get_modality_lengths()
    #         sampler = SplitModalitySampler(
    #             dataset,
    #             modality_lengths,
    #             global_batch_size=self.global_batch_size,
    #             num_replicas=overwatch.world_size(),
    #             rank=overwatch.rank(),
    #             seed=seed,
    #             drop_last=False,
    #         )

    #     else:
    #         sampler = DistributedSampler(
    #             dataset,
    #             num_replicas=overwatch.world_size(),
    #             rank=overwatch.rank(),
    #             shuffle=True,
    #             seed=seed,
    #             drop_last=False,
    #         )

    #     # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=self.per_device_batch_size,
    #         sampler=sampler,
    #         collate_fn=collator,
    #         num_workers=2,
    #         worker_init_fn=self.worker_init_fn,
    #     )

    #     # === Train ===
    #     status = metrics.get_status()
        
    #     # Outer progress bar for epochs
    #     epoch_progress = tqdm(
    #         range(self.epochs), desc=f"Epochs", leave=True, disable=not overwatch.is_rank_zero(), position=0
    #     )
    #     for epoch in epoch_progress:
    #         self.vlm.train()
    #         sampler.set_epoch(epoch)
    #         # Zero-Gradients (just in case)
    #         self.optimizer.zero_grad()

    #         # Inner progress bar for steps within each epoch
    #         step_progress = tqdm(
    #             enumerate(dataloader), desc=f"Epoch {epoch+1}/{self.epochs} - {status}",
    #             leave=False, disable=not overwatch.is_rank_zero(), total=len(dataloader), position=1
    #         )
    #         # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
    #         #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
    #         for train_idx, batch in step_progress:
    #             # [Contract] self.vlm.forward() must automatically compute `loss` and return!
    #             with torch.autocast(
    #                 "cuda",
    #                 dtype=self.mixed_precision_dtype,
    #                 enabled=self.enable_mixed_precision_training,
    #             ):
    #                 output: CausalLMOutputWithPast = self.vlm(
    #                     input_ids=batch["input_ids"],
    #                     attention_mask=batch["attention_mask"],
    #                     pixel_values=batch["pixel_values"],
    #                     labels=batch["labels"],
    #                     multimodal_indices=batch["multimodal_indices"],
    #                 )
    #                 loss = output.loss

    #             # Commit Loss (Prior to Gradient Accumulation Normalization)
    #             metrics.commit(loss=loss)

    #             # Normalize Loss to account for Gradient Accumulation --> Backward!
    #             # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
    #             #             because in general, each batch has a *different number of masked out tokens* (because
    #             #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
    #             #
    #             #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
    #             #             the "correct" implementation, without adding extra complexity.
    #             #
    #             # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
    #             #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
    #             #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
    #             #   someone to PR and fix this (and I'd greatly appreciate it!!!)
    #             normalized_loss = loss / self.grad_accumulation_steps
    #             normalized_loss.backward()

    #             # Step =>> Only if Done w/ Gradient Accumulation
    #             if (train_idx + 1) % self.grad_accumulation_steps == 0:
    #                 metrics.commit(update_step_time=True)

    #                 # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
    #                 self.clip_grad_norm()

    #                 # Optimizer & LR Scheduler Step
    #                 self.optimizer.step()
    #                 self.lr_scheduler.step()
    #                 self.optimizer.zero_grad()

    #                 # Push Metrics
    #                 metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
    #                 status = metrics.push()

    #                 # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
    #                 if self.max_steps is not None and metrics.global_step >= self.max_steps:
    #                     self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
    #                     dist.barrier()

    #                     return

    #                 # Update Progress Bars
    #                 step_progress.set_description(f"Epoch {epoch+1}/{self.epochs} - {status}")

    #         # Update epoch progress bar after finishing all steps in this epoch
    #         epoch_progress.set_description(f"Epochs - {status}")
            
    #         # Save checkpoint at end each epoch (if `self.max_steps` is None)
    #         if self.max_steps is None:
    #             self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
    #             dist.barrier()
