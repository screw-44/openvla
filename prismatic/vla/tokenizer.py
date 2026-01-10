from abc import abstractmethod, ABC
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Any, Type, Tuple, Dict

import torch
import numpy as np

from torchvision import transforms
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.vision.base_vision import ImageTransform
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.vlms.prismatic import IGNORE_INDEX
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


TRAJECTORY_CONVERTER_REGISTRY = {}


def register_trajectory_converter(name: str):
    def decorator(cls):
        TRAJECTORY_CONVERTER_REGISTRY[name] = cls
        return cls

    return decorator


# 不同的数据集有不同的表征方式. libero是x,y,z,yaw,pitch,row + gripper夹取。
# TODO: 未来可以实现更复杂的TrajectoryConverter，比如VQ-VAE编码+离散化 / 多个token代表一个浮点数等方式。 / 多个token代表一个空间中的点。
class BaseTrajectoryConverter(ABC):
    """
    其实是转换 单个浮点数(或者是轨迹上的点) 到 离散文本 的过程。
    - 直接映射一个或者多个token表示一个浮点数
    - VQ-VAE编码+离散化
    - 不转换成离散文本也可以，用mlp+l1回归的方式。
    """

    def __init__(self):
        self.n_dims = 0

    @abstractmethod
    def encode_trajectory_to_token_ids(self, trajectory: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode_text_ids_to_trajectory(self, texts: str) -> np.ndarray:
        pass


@register_trajectory_converter("value_textualize")
class ValueTextualizeTC(BaseTrajectoryConverter):
    """将[-1,1]浮点数离散化为bin索引，映射到vocab_size-512~vocab_size-256的token（避免与EOS token 50256冲突）"""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, n_bins: int = 256, n_dims: int = 2
    ):
        self.tokenizer = tokenizer
        self.n_bins = n_bins
        self.n_dims = n_dims
        self.bins = np.linspace(-1.0, 1.0, n_bins)  # 离散化的分界线
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # 每个bin的中心点
        self.action_token_start = self.tokenizer.vocab_size - 512  # 动作token起始位置

    def encode_trajectory_to_token_ids(self, trajectory: np.ndarray) -> np.ndarray:
        # print("trajectory:", trajectory)
        assert (
            self.n_dims == trajectory.shape[1]
        ), f"维度不匹配：期望{self.n_dims}，得到{trajectory.shape[1]}"
        trajectory = np.clip(np.asarray(trajectory, dtype=np.float32), -1.0, 1.0)
        # digitize返回[1, n_bins]，-1转换为[0, n_bins-1]用于索引bin_centers
        bin_indices = np.digitize(trajectory, self.bins) - 1
        # 映射到token id: [vocab_size-512, vocab_size-256)
        token_ids = (self.action_token_start + bin_indices.flatten()).astype(int)
        return token_ids

    def decode_text_ids_to_trajectory(self, text_ids: np.ndarray) -> np.ndarray:
        text_ids = text_ids[:-1]  # 移除EOS token
        # 反向映射：token_id -> bin索引，并clip到有效范围
        bin_indices = np.clip(
            text_ids - self.action_token_start, 0, len(self.bin_centers) - 1
        )
        continuous = self.bin_centers[bin_indices]  # 用bin中心值作为连续估计
        return continuous.reshape(-1, self.n_dims)
    
@register_trajectory_converter("abs_aff_bspline_textualize")
class AbsAffBsplineTextualizeTC(BaseTrajectoryConverter):
    """用分位数量化B样条控制点，映射到vocab_size-512~vocab_size-256的token（避免与EOS token 50256冲突）"""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, n_bins: int = None, n_dims: int = None,
        edges_path: str = None
    ):
        self.tokenizer = tokenizer
        self.n_bins = 512  # 强制设为512
        self.n_dims = 8    # 7+knot
        self.action_token_start = self.tokenizer.vocab_size - self.n_bins - 256  # 动作token起始位置
        
        # 加载分位数 edges (shape: 6, 513)
        if edges_path is None:
            edges_path = Path(__file__).parent / "libero_dim_bin_distribution.npy"
        
        self.edges = np.load(edges_path)  # shape: (6, n_bins+1)
        assert self.edges.shape == (6, self.n_bins + 1), f"edges shape error: {self.edges.shape}"
        # 计算每个维度的 bin center
        self.bin_centers = np.array([
            (self.edges[d, :-1] + self.edges[d, 1:]) / 2.0 for d in range(6)
        ])
        overwatch.info(f"✓ Loaded quantile edges from {edges_path}")

    def encode_trajectory_to_token_ids(self, trajectory: np.ndarray) -> np.ndarray:
        bspline_trajectory, clamp_trajectory, knot_trajectory = trajectory[:, :6], trajectory[:, 6], trajectory[:, 7]
        
        # 编码 bspline: 用 searchsorted 找到分位数 bin 索引
        bspline_traj_token_ids = np.zeros_like(bspline_trajectory, dtype=int)
        for d in range(6):
            bin_idx = np.searchsorted(self.edges[d], bspline_trajectory[:, d], side='right') - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            bspline_traj_token_ids[:, d] = self.action_token_start + bin_idx

        # 编码 clamp
        clamp_traj_token_ids = self.action_token_start + (clamp_trajectory + 1) * (self.n_bins - 1) // 2

        # 编码 knot
        knot_trajectory_token_ids = self.action_token_start + knot_trajectory
        
        # 拼接所有 token
        traj_token_ids = np.concatenate([
            bspline_traj_token_ids, 
            clamp_traj_token_ids.reshape(-1, 1), 
            knot_trajectory_token_ids.reshape(-1, 1)
        ], axis=1).flatten().astype(int)

        return traj_token_ids

    def decode_text_ids_to_trajectory(self, text_ids: np.ndarray) -> np.ndarray:
        # 移除EOS token（如有）
        text_ids = text_ids[:-1]
        n = text_ids.shape[0] // 8
        # 还原为 (n, 8)
        tokens = text_ids.reshape(n, 8)
        
        # 还原 bspline: 用 bin center 来近似
        bspline_token_ids = tokens[:, :6]
        bspline = np.zeros((n, 6), dtype=np.float32)
        for d in range(6):
            bin_idx = np.clip(bspline_token_ids[:, d] - self.action_token_start, 0, self.n_bins - 1)
            bspline[:, d] = self.bin_centers[d, bin_idx]
        
        # 还原 clamp
        clamp_token_ids = tokens[:, 6]
        clamp_value = (clamp_token_ids - self.action_token_start) * 2 / (self.n_bins - 1) - 1
        
        # 还原 knot
        knot_token_ids = tokens[:, 7]
        knot = knot_token_ids - self.action_token_start
        
        # 拼回 (n, 8)
        trajectory = np.concatenate([bspline, clamp_value[:, None], knot[:, None]], axis=1)
        return trajectory
        


# 使用dataclass方便管理参数
@dataclass
class VlaTokenizer:
    """
    从dataset的batch中转换成prismatic的训练输入输出格式，如下： \n
    batch["input_ids"]:
    batch["attention_mask"]:
    batch["pixel_values"]:
    batch["labels"]:
    """

    trajectory_converter: BaseTrajectoryConverter  # 注意这个的trajectory tokenizer应该返回的是文本，而不是token ids（后续还需要编码）
    base_tokenizer: PreTrainedTokenizerBase  # huggingface Transformers的tokenizer类。会将文本分词/编码到id/添加特殊token
    prompt_builder_fn: Type[PromptBuilder]

    def tokenize_input(self, batch: Dict[str, Any]) -> dict:
        # 定义vla的conversation prompt
        lang = batch["language"].lower().strip()
        prompt_builder = self.prompt_builder_fn("openvla")

        # Step1: 添加human turn，获取prompt（含"In: ...\nOut: "但不含action）
        prompt_builder.add_turn(
            "human", f"What action should the robot take to {lang}?"
        )
        prompt = prompt_builder.get_prompt()  # 注意gpt2默认不添加eos tokens
        prompt_ids = self.base_tokenizer(prompt)["input_ids"]

        return dict(
            pixel_values={"cam1": batch["cam1"], "cam2": batch["cam2"]},
            prompt_ids=prompt_ids,
        )

    def tokenize_batch(self, batch: Dict[str, Any], train: bool = True) -> dict:
        """Convert raw batch from dataset into model-ready inputs/labels."""
        # TODO： 未来这里加入对数据集的gripper的滤波/裁剪等预处理操作
        inputs = self.tokenize_input(batch)
        action_tokens = self.trajectory_converter.encode_trajectory_to_token_ids(
            batch["trajectory"]
        )
        input_ids = np.hstack(
            [inputs["prompt_ids"], action_tokens, [self.base_tokenizer.eos_token_id]]
        )

        # Step6: 只保留action部分的labels，其余设为IGNORE_INDEX
        labels = input_ids.copy()
        prompt_ids_length = len(inputs["prompt_ids"])
        labels[:prompt_ids_length] = IGNORE_INDEX

        # overwatch.info(f"input_ids: {prompt_ids}")
        # overwatch.info(f"full input_ids: {input_ids}")
        # overwatch.info(f"labels: {labels}")
        return dict(
            pixel_values=inputs["pixel_values"],
            input_ids=torch.tensor(input_ids),
            labels=torch.tensor(labels),
            dataset_names=batch["dataset_names"],
            prompt_ids_length=torch.tensor(prompt_ids_length),
        )
