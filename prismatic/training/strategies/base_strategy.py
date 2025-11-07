"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf.run import BaseStrategyConfig
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
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

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        validate_cfg: Optional[BaseStrategyConfig] = None,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
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
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # print(f"predicted action length: {action_preds.shape}, gt action length: {action_gt.shape}, mask shape: {mask.shape}")
                # print(f"mask true length is {mask[0].sum()}")
                # -------------------------------------------------------
                # predicted action length: torch.Size([64, 321]), gt action length: torch.Size([64, 321]), mask shape: torch.Size([64, 321])
                # predicted action length: torch.Size([64, 321]), gt action length: torch.Size([64, 321]), mask shape: torch.Size([64, 321])
                # mask true length is 234

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                # print(f"continous_actions_pred - continous_actions_gt [0:5]: {continuous_actions_pred[0:5] - continuous_actions_gt[0:5]}")

                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Do Validation if specified
                if validate_cfg is not None and validate_cfg.is_validate:
                    if metrics.global_step % validate_cfg.validate_interval == 0:
                        overwatch.info(f"Running Validation at Step {metrics.global_step}...")
                        self.run_vla_testing(
                            vla_dataset, # TODO： 添加训练数据集/测试数据集/验证数据集的区分
                            collator,
                            action_tokenizer,
                            test_cfg=validate_cfg,
                        )
                        overwatch.info("Done with Validation.")

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

    # === VLA Testing ===

    def run_vla_testing(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        test_cfg: BaseStrategyConfig,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        import numpy as np
        from PIL import Image
        import cv2
        from Aff_benchmark.utils.rotation import SE3Converter
        from Aff_benchmark.visualization.aff_2d import draw_aff_on_image
        if not overwatch.is_rank_zero(): # 只需要在rank0上进行测试
            return

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Testing ===
        self.vlm.eval()

        # 保存的数据
        save_data_all = {}

        # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
        #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
        #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
        for i, batch in tqdm(enumerate(dataloader)):
            # 因为dataloader是无限循环的，所以需要手动控制测试长度（读取实际的数据集长度）
            if i > len(vla_dataset) or (i >= test_cfg.test_data_length and test_cfg.test_data_length > 0):
                break
            
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
            save_data_batch = {}
            # To compute action token accuracy, we need to identify the locations of the action tokens
            # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
            # insert `self.vlm.vision_backbone.num_patches` at index 1.
            #
            # Computing `action_prediction_accuracy` is then pretty straightforward:
            #   1) Extract "aligned" predictions & labels
            #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
            #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
            #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
            action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # print(f"predicted action length: {action_preds.shape}, gt action length: {action_gt.shape}, mask shape: {mask.shape}")
            # print(f"mask true length is {mask[0].sum()}")
            # -------------------------------------------------------
            # predicted action length: torch.Size([64, 321]), gt action length: torch.Size([64, 321]), mask shape: torch.Size([64, 321])
            # predicted action length: torch.Size([64, 321]), gt action length: torch.Size([64, 321]), mask shape: torch.Size([64, 321])
            # mask true length is 234

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            save_data_batch["action_accuracy"] = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            # print(f"continous_actions_pred - continous_actions_gt [0:5]: {continuous_actions_pred[0:5] - continuous_actions_gt[0:5]}")
            save_data_batch["object_pose_pred"] = continuous_actions_pred[0].cpu().numpy()
            save_data_batch["aff_pred"] = continuous_actions_pred[1:].cpu().numpy()
            save_data_batch["object_pose_gt"] = continuous_actions_gt[0].cpu().numpy()
            save_data_batch["aff_gt"] = continuous_actions_gt[1:].cpu().numpy()
            save_data_batch["object_pose_l1_loss"] = torch.nn.functional.l1_loss(continuous_actions_pred[0:1], continuous_actions_gt[0:1])
            save_data_batch["aff_l1_loss"] = torch.nn.functional.l1_loss(continuous_actions_pred[1:], continuous_actions_gt[1:])

            # 如果开启保存目录，就进行可视化保存
            if test_cfg.test_save_dir is not None:
                vis_save_dir = test_cfg.test_save_dir / "visualizations"
                vis_save_dir.mkdir(parents=True, exist_ok=True) # 
                visual_img = batch["pixel_values"][0].cpu().numpy() # batch["pixel_values"].shape = [batchsize, 3, 224, 224]
                # print(visual_img.min().item(), " ", visual_img.max().item())
                # 调整通道顺序 (C, H, W) → (H, W, C)
                visual_img = np.transpose(visual_img, (1, 2, 0))      # (224, 224, 3) (H, W, 3) 
                visual_img = (visual_img + 1) / 2
                visual_img = (visual_img * 255).astype(np.uint8)
                # print(visual_img.min().item(), " ", visual_img.max().item())
                visual_img = cv2.resize(visual_img, (1280, 720))
                # 读取相机内参 ob camera
                camera_K = np.array(
                    [[685.95849609375, 0.0, 644.1708984375],
                     [0.0, 686.1210327148438, 362.3411560058594],
                     [0.0, 0.0, 1.0]]
                )
                trajectory_pred = SE3Converter.d9_to_se3(save_data_batch["aff_pred"])
                pose_6d_pred = SE3Converter.d9_to_se3(save_data_batch["object_pose_pred"])
                trajectory_gt = SE3Converter.d9_to_se3(save_data_batch["aff_gt"])
                # print("trajectory_gt shape:", trajectory_gt.shape)
                pose_6d_gt = SE3Converter.d9_to_se3(save_data_batch["object_pose_gt"])
                # print("object pose shape:", pose_6d_gt.shape)

                # 如果是object centircs,这里trajectory的key是object
                trajectory_key = "object"
                visualized_img_pred = draw_aff_on_image(
                    image=visual_img,
                    object_pose=pose_6d_pred,
                    trajectory=trajectory_pred,
                    K=camera_K,
                    object_gt_pose=pose_6d_gt, # 使用gt的物体位姿作为参考
                )

                visualized_img_gt = draw_aff_on_image(
                    image=visual_img,
                    object_pose=pose_6d_gt,
                    trajectory=trajectory_gt,
                    K=camera_K,
                    object_gt_pose=pose_6d_gt, # 使用gt的物体位姿作为参考
                )

                # === 左右拼接图像并添加文字标签 ===
                # 左右拼接图像
                combined_img = np.hstack([visualized_img_pred, visualized_img_gt])
                
                # 在图像顶部添加文字标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_color = (255, 255, 255)  # 白色
                font_thickness = 3
                
                # 计算文字位置
                text_pred = "PRED"
                text_gt = "GT"
                text_size_pred = cv2.getTextSize(text_pred, font, font_scale, font_thickness)[0]
                text_size_gt = cv2.getTextSize(text_gt, font, font_scale, font_thickness)[0]
                
                # 在左半部分（pred）顶部居中添加"PRED"
                x_pred = (1280 - text_size_pred[0]) // 2
                y_pred = 40  # 距离顶部40像素
                cv2.putText(combined_img, text_pred, (x_pred, y_pred), font, font_scale, font_color, font_thickness)
                
                # 在右半部分（gt）顶部居中添加"GT"
                x_gt = 1280 + (1280 - text_size_gt[0]) // 2
                y_gt = 40  # 距离顶部40像素
                cv2.putText(combined_img, text_gt, (x_gt, y_gt), font, font_scale, font_color, font_thickness)

                # # 保存原始单独图像
                # cv2.imwrite(str(vis_save_dir / f"test_{i:05d}_pred.png"), visualized_img_pred)
                # cv2.imwrite(str(vis_save_dir / f"test_{i:05d}_gt.png"), visualized_img_gt)
                
                combined_img = cv2.resize(combined_img, None, fx=0.3, fy=0.3)  #压缩图像大小到接近到224的地方
                # 保存拼接后的图像
                cv2.imwrite(str(vis_save_dir / f"test_{i:05d}_combined.jpg"), combined_img)



            # 保存数据到整体的字典中
            save_data_all[i] = save_data_batch

        # === 循环结束后统计 object_pose_l1_loss 的统计信息 ===
        if len(save_data_all) > 0:
            # 提取所有batch的 object_pose l1 loss
            obj_l1_losses, aff_l1_losses = [], []
            for batch_data in save_data_all.values():
                obj_l1_loss_value = batch_data["object_pose_l1_loss"]
                aff_l1_loss_value = batch_data["aff_l1_loss"]
                # 如果是tensor，转为float
                if hasattr(obj_l1_loss_value, 'item'):
                    obj_l1_losses.append(obj_l1_loss_value.item())
                else:
                    obj_l1_losses.append(float(obj_l1_loss_value))
                if hasattr(aff_l1_loss_value, 'item'):
                    aff_l1_losses.append(aff_l1_loss_value.item())
                else:
                    aff_l1_losses.append(float(aff_l1_loss_value))
            
            # 转为tensor便于计算统计量
            obj_l1_losses_tensor = torch.tensor(obj_l1_losses)
            aff_l1_losses_tensor = torch.tensor(aff_l1_losses)

            print("="*60)
            print("object_pose_l1_loss 统计信息:")
            print(f"  总样本数: {len(obj_l1_losses)}")
            print(f"  最小值 (Min):     {obj_l1_losses_tensor.min().item():.6f}")
            print(f"  最大值 (Max):     {obj_l1_losses_tensor.max().item():.6f}")
            print(f"  均值 (Mean):      {obj_l1_losses_tensor.mean().item():.6f}")
            print(f"  标准差 (Std):     {obj_l1_losses_tensor.std().item():.6f}")
            print("\naff_l1_loss 统计信息:")
            print(f"  总样本数: {len(aff_l1_losses)}")
            print(f"  最小值 (Min):     {aff_l1_losses_tensor.min().item():.6f}")
            print(f"  最大值 (Max):     {aff_l1_losses_tensor.max().item():.6f}")
            print(f"  均值 (Mean):      {aff_l1_losses_tensor.mean().item():.6f}")
            print(f"  标准差 (Std):     {aff_l1_losses_tensor.std().item():.6f}")
            print("="*60)
        else:
            print("Warning: No data collected for statistics!")

        