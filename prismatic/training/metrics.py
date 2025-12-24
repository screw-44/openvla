"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging.
Now uses trackio (imported as trackio) for experiment tracking.
"""

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import pandas as pd
import numpy as np
import torch
import trackio

from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.vlms.vla import VLA
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class VLAMetrics:
    def __init__(
        self,
        run_id: str,
        hparams: Dict[str, Any],
        group: str,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
        project: str = "vla-training",
    ) -> None:
        trackio.init(project=project, name=run_id, config=hparams, group=group)

        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch

        self.log_freq = 2  # 每记录10次才log一次
        self.detail_log_freq = 20  # 每100次才详细的log一次

    def log_pro(
        self,
        output: CausalLMOutputWithPast,
        input: Dict[str, torch.Tensor],  # input
        vla: VLA,  # PrismaticVLM model
        lr: float,
    ) -> None:
        """
        Compute all action prediction metrics from model output and batch labels.

        Handles variable-length sequences by inferring action start positions from labels,
        extracting action logits safely, and computing accuracy/L1 metrics.

        Args:
            output: CausalLMOutputWithPast containing logits [B, L, V]
            batch: Dict with 'labels' [B, L], 'pixel_values', etc.
            vla: PrismaticVLM model (for trajectory_converter)
            global_step: Current training step (for logging)
        """
        # 按照一定的频率来进行记录（这里比较costly，不全部记录）
        # if self.global_step % self.log_freq != 0:
        #     return

        # print("keys:", input.keys())

        labels = input["labels"].to("cpu").numpy()
        prompt_ids_length = input["prompt_ids_length"].to("cpu").numpy()
        predicts = output.logits.argmax(dim=2).to("cpu").numpy()
        batch_size = predicts.shape[0]
        # print("label shape", batch["labels"].shape)
        # print("input shape:", batch["input_ids"].shape)
        # print("output logits shape:", output.logits.shape) # NOTE：这里输出中有image tokens，所以会远远长于输入

        total_correct, total_tokens = 0, 0  # token级别的准确率
        full_cont_l1_loss, t0_cont_l1_loss = 0, 0  # 解码后连续action的l1距离
        table_inputs, table_preds, table_gts = [], [], []

        for i in range(batch_size):
            start_idx = int(prompt_ids_length[i])
            # GT: action tokens in labels (from start_idx onwards, excluding padding/eos)
            gt = labels[i, start_idx:]
            gt = gt[gt != -100]  # Remove IGNORE_INDEX padding
            # Pred: 注意这里还有image的tokens，所以长度不match。所以很难找到对应的输入
            # 同时看上去pred比输入整体往左偏移了1个token，（后面pad了一个token）。自回归的范式吧
            padding_length = (labels[i, start_idx:] == -100).sum()
            # print("padding length", padding_length, " gt length: ", len(gt))
            pred_start_idx, pred_end_idx = (
                predicts.shape[1] - len(gt) - padding_length - 1,
                predicts.shape[1] - padding_length - 1,
            )
            pred = predicts[i, pred_start_idx:pred_end_idx]
            print(f"pred: {pred}, gt: {gt}")

            # 1. Token-level accuracy
            total_correct += (pred == gt).sum()
            total_tokens += len(gt)
            # 2. Continuous trajectory L1 loss
            cont_pred = vla.trajectory_converter.decode_text_ids_to_trajectory(pred)
            cont_gt = vla.trajectory_converter.decode_text_ids_to_trajectory(gt)
            full_cont_l1_loss += torch.nn.functional.l1_loss(
                torch.as_tensor(cont_gt), torch.as_tensor(cont_pred)
            ).item()
            cont_pred_t0, cont_gt_t0 = cont_pred[0], cont_gt[0]
            t0_cont_l1_loss += torch.nn.functional.l1_loss(
                torch.as_tensor(cont_pred_t0), torch.as_tensor(cont_gt_t0)
            ).item()

            table_inputs.append(input["input_ids"][i, :start_idx])
            table_preds.append(pred)
            table_gts.append(gt)

        # print("table_pred:", table_preds[0:5])
        # print("table_gts:", table_gts[0:5])

        token_precision = total_correct / total_tokens
        # print("token precision:", token_precision)
        full_cont_l1_loss /= batch_size
        t0_cont_l1_loss /= batch_size
        # print("full_l1_loss", full_cont_l1_loss)
        # print("t0_cont_l1_loss", t0_cont_l1_loss)

        prefix = "VLA Train"
        trackio.log(
            {
                f"{prefix}/Loss": output.loss.item(),
                f"{prefix}/Token Precision": token_precision,
                f"{prefix}/Full Trajectory L1 Loss": full_cont_l1_loss,
                f"{prefix}/T0 L1 Loss": t0_cont_l1_loss,
                f"{prefix}/Learning Rate": lr,
            },
            step=self.global_step,
        )

        # ===== 这里开始是记录image和table的数据, 控制频率来实现加速 =====
        if self.global_step % self.detail_log_freq != 0:
            return

        # Decode input_ids 为文本（只处理第一个样本）, table_inputs[0] 是 input token IDs
        prompt_tokens = table_inputs[0]
        # print("lables[0]", input["input_ids"][0])
        # print("prompt_tokens", prompt_tokens)
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.tolist()
        prompt_text = vla.llm_backbone.tokenizer.decode(
            prompt_tokens, skip_special_tokens=True
        )
        table_inputs_decoded = [prompt_text]

        # 处理图像数据（pixel_values 是 dict 格式，key为 'cam1', 'cam2'）
        # 取两个摄像头的第一个样本并拼接
        pixel_values = input["pixel_values"]

        # 取 cam1 和 cam2 的第一个样本 [C, H, W]，并在width维度拼接
        img_tensor = torch.cat(
            [pixel_values["cam1"][0], pixel_values["cam2"][0]], dim=2
        )  # [C, H, W*2]
        # 反向归一化到[0,255]并resize为256x256
        img_np = img_tensor.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
        img_np = ((img_np*0.5 + 1) * 255).clip(0, 255).astype(np.uint8)

        # 创建DataFrame记录
        df = pd.DataFrame(
            {
                "prompt": table_inputs_decoded,
                "pred": [str(table_preds[0])[:100]] if table_preds else ["[无预测]"],
                "gt": [str(table_gts[0])[:100]] if table_gts else ["[无真值]"],
            }
        )

        trackio.log(
            {
                f"{prefix}/训练输入输出记录": trackio.Table(dataframe=df),
                f"{prefix}/图像样本": trackio.Image(img_np),
            }
        )

    def get_status(self, loss: Optional[torch.Tensor] = None, lr: float = 0) -> str:
        return f"=>> [Epoch {self.epoch:03d}] G_Step {self.global_step:06d} =>> Lr:{lr:.6f} - Loss:{loss:.4f}"

    def finalize(self) -> str:
        trackio.finish()
