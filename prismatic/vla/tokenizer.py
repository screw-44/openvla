from abc import abstractmethod, ABC
from dataclasses import dataclass
from os import path
from typing import Any, Type, Tuple, Dict

import torch
import numpy as np

from torchvision import transforms
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.vision.base_vision import ImageTransform
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.vlms.prismatic import IGNORE_INDEX


TRAJECTORY_CONVERTER_REGISTRY = {}

def register_trajectory_converter(name: str):
    def decorator(cls):
        TRAJECTORY_CONVERTER_REGISTRY[name] = cls
        return cls
    return decorator

# 不同的数据集有不同的表征方式. libero是x,y,z,yaw,pitch,row + gripper夹取。
# 但是别的数据集是不一样的，比如说智元添加了双臂+底盘。 
class BaseTrajectoryConverter(ABC):
    """
        其实是转换 单个浮点数(或者是轨迹上的点) 到 离散文本 的过程。
        - 直接映射一个或者多个token表示一个浮点数
        - VQ-VAE编码+离散化
        - 不转换成离散文本也可以，用mlp+l1回归的方式。
    """
    @abstractmethod
    def encode_trajectory_to_texts(self, trajectory: np.ndarray) -> str:
        pass

    @abstractmethod
    def decode_text_ids_to_trajectory(self, texts: str) -> np.ndarray:
        pass

@register_trajectory_converter("value_textualize")
class ValueTextualizeTC(BaseTrajectoryConverter):
    """
        最简单的离散化方式：直接将浮点数转换成字符串表示，并用空格分隔开。TC代表Trajectory Converter。
        输入应该均一化过，都在[-1, 1]范围内。
        之前是一个dim，后一个dim，先后顺序拼接成字符串。（应该是时间维度上展开）
        例如：trajectory = [[0.1, 0.2], [0.3, 0.4]] => "0.1 0.3" 然后 "0.2 0.4"

        如果你想保留梯度进行端到端训练，Gumbel-Softmax 或 Straight-Through Estimator (STE) 这类可微分的离散化技术，而不是当前的硬离散化方案。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, n_bins: int = 256, n_dims: int = 2):
        self.tokenizer = tokenizer
        self.n_bins = n_bins
        self.n_dims = n_dims # 输出动作的维度
        self.bins = np.linspace(-1.0, 1.0, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.trajectory_token_begin_idx: int = int(self.tokenizer.vocab_size - (n_bins + 1))

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    def encode_trajectory_to_texts(self, trajectory: np.ndarray) -> str:
        assert self.n_dims == trajectory.shape[1], f"输入轨迹的维度应该是{self.n_dims}，但是得到的是{trajectory.shape[1]}"
        # trajectory是一个[points_num, dim]的numpy数组
        trajectory = np.clip(trajectory, -1.0, 1.0)
        trajectory = np.transpose(trajectory)  # 转置成[dim, points_num] (预测的时候按照时间维度展开)
        trajectory = np.digitize(trajectory, self.bins)  # 离散化, 返回的是索引+1, [1, n_bins]
        # 通过减法，决定映射到词表最不常用的256个token上 （保持语言能力，最小化干扰）.注意要flatten成一维向量
        return self.tokenizer.decode(list(self.tokenizer.vocab_size - trajectory.flatten()))

    def decode_text_ids_to_trajectory(self, text_ids: np.ndarray) -> np.ndarray:
        # llm输出就是text_ids, 所以不需要从text进行转换。直接减法就能映射会0-n_bins离散值
        discretized_trajectory = self.tokenizer.vocab_size - text_ids
        # 映射回bin_centers的id（主要是clip到0-n_bins-1范围内，避免越界）
        discretized_trajectory = np.clip(discretized_trajectory - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        # look up table找回去就好（用bin_centers，他们觉得比bins精度更好，最小化精度误差）
        continuous_trajectory = self.bin_centers[discretized_trajectory]
        # 变回矩阵形式，并回到原本的dim作为维度(注意要transpose过来，所以是先n_dims，再-1)
        continuous_trajectory = continuous_trajectory.reshape(self.n_dims, -1).T
        return continuous_trajectory


# TODO: 未来可以实现更复杂的TrajectoryConverter，比如VQ-VAE编码+离散化 / 多个token代表一个浮点数等方式。 / 多个token代表一个空间中的点。


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
    trajectory_converter: BaseTrajectoryConverter # 注意这个的trajectory tokenizer应该返回的是文本，而不是token ids（后续还需要编码）
    base_tokenizer: PreTrainedTokenizerBase   # huggingface Transformers的tokenizer类。会将文本分词/编码到id/添加特殊token
    image_transform: ImageTransform          # 这个image transform中应该包含resize的事情
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    default_image_resolution: Tuple[int, int] = (224, 224)

    def tokenize_batch(self, batch:  Dict[str, Any], train: bool = True) -> dict:
        """Convert raw batch from dataset into model-ready inputs/labels."""
        # 定义vla的conversation prompt、
        lang = batch["language"].lower().strip()
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.trajectory_converter.encode_trajectory_to_texts(batch['trajectory'])}
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn['from'], turn['value']) # 添加对话轮次，积累格式化文本

        # Causal Langauge Modeling的训练方式：根据前面一个token预测下一个token input_ids[i]为输入，labels[i+1]为输出
        # get_prompt()反馈多论对话后完整*格式化*文字字符串（和VLM模型强绑定）。
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # 目前只是一个图像输入，可以修改成多个图像输入 (现在实现传入已经是tensor了)
        # img = Image.fromarray(batch['rgb'])
        # img = img.resize(self.default_image_resolution)
        # TODO： 未来这里加入对数据集的gripper的滤波/裁剪等预处理操作
        # 这里应该包含resize等操作。注意lerobot数据集默认会转换成[0, 1]的torch.Tensor
        # print("image transform is :", self.image_transform)
        # TODO： 由于目前lerobot数据集直接输出的是tensor格式的图像，所以这里需要修改image_transform的输入类型。暂时放弃使用，未来需要修改：
        # image transform is : Compose(
        # LetterboxPad(padding_fill_value=(127, 127, 127))
        # Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
        # CenterCrop(size=(224, 224))
        # ToTensor()
        # Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000])))
        
        vision_transform = transforms.Compose([
            transforms.ToPILImage(),
            self.image_transform,
        ])

        # print("min value is: ", torch.min(batch['rgb']), " max value is: ", torch.max(batch['rgb']))
        # print("mean value is: ", torch.mean(batch['rgb']), " std value is: ", torch.std(batch['rgb']))
        # exit()
        # in value is:  tensor(0.)  max value is:  tensor(0.9490):: 0.000000:   0%|   | 0/273465 [00:00<?, ?it/s]
        # mean value is:  tensor(0.2715)  std value is:  tensor(0.1609)

        pixel_values = vision_transform(batch['rgb'] * 255)
        #pixel_values = batch['rgb']

        # [Openvla: CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # 所以我们不计算输入的语言部分的loss，只计算最后的动作token部分的loss
        # print("trajectory shape is: ", batch['trajectory'].shape)
        num_trajectory_tokens = batch['trajectory'].shape[0] * batch['trajectory'].shape[1] # 时间步数*维度数
        labels[: -(num_trajectory_tokens+1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_names=batch["dataset_names"])


# ============================================================================
# Dummy Tokenizer for Visualization and Debugging
# ============================================================================
# 用于可视化和调试的简化 Tokenizer，避免完整初始化 VlaTokenizer 的复杂依赖
# （image_transform, base_tokenizer 等）
# ============================================================================

class DummyTokenizer:
    """
    用于可视化和调试的 Dummy Tokenizer
    
    提供与 VlaTokenizer 相同的接口，但跳过实际的 tokenization 过程。
    主要用于：
    1. trajectory_compression 的可视化和测试
    2. 数据集加载的调试
    3. 避免加载大型预训练模型和图像转换器
    
    使用方式：
        dummy_tokenizer = DummyTokenizer()
        dataset = MyLeRobotDataset(
            repo_id="HuggingFaceVLA/libero",
            tokenizer=dummy_tokenizer,
            trajectory_compression=BiningTrajectoryCompression(),
            task_ids=[0, 1, 2]
        )
    """
    
    def __init__(self):
        """初始化 Dummy Tokenizer，无需任何依赖"""
        pass
    
    def tokenize_batch(self, batch: Dict[str, Any], train: bool = True) -> dict:
        """
        简化的 tokenize_batch 实现
        
        Args:
            batch: 包含 'rgb', 'language', 'trajectory', 'dataset_names' 的字典
            train: 是否为训练模式（此参数在 dummy 版本中被忽略）
        
        Returns:
            包含原始数据的字典，供调试使用
        """
        return {
            'rgb': batch['rgb'],
            'language': batch['language'],
            'trajectory': batch['trajectory'],
            'dataset_names': batch['dataset_names'],
            # 添加 dummy 的 tokenization 字段（供兼容性使用）
            'input_ids': None,
            'labels': None,
            'pixel_values': None,
        }

