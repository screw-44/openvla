"""
Strategy的Config文件

这个文件用于存储test和validate的configutreation的文件，其both是TrainConfig的Has-A关系。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Tuple
from enum import Enum, unique
from draccus import ChoiceRegistry

from .vla import VLAConfig, VLARegistry


# == Define Runtime Strategy Configuration for Validation & Testing == #
# 在 Python 的 dataclass 中，有一个重要的规则：子类中不能在有默认值的字段后面定义没有默认值的字段。
# NOTE： 如果子类有默认数值，夫类一定要有默认数值，否则会报错。
@dataclass
class BaseStrategyConfig(ChoiceRegistry):
    """所有选项的基, 不应该被初始化."""
    # Testing Parameters
    is_validate: bool = False                                    # Whether we are in validation mode
    is_test: bool = False                                       # Whether we are in testing mode
    test_save_dir: Optional[Path] = None                       # Directory to save test results

@dataclass
class PureTrain(BaseStrategyConfig):
    """子类，代表不同的训练strategy设定（输入参数） Configuration for pure training without validation or testing."""
    test_config_id: str = "default"                         # Unique identifier
    is_validate: bool = False                               # Disable validation during training
    is_test: bool = False                                  # Disable testing mode
    test_save_dir: Optional[Path] = None                   # No test results to save

@dataclass
class TrainWithValidation(BaseStrategyConfig):
    """子类，代表不同的训练strategy设定（输入参数） Configuration for training with validation enabled."""
    test_config_id: str = "train-with-validation"                   # Unique identifier
    is_validate: bool = True                                        # Enable validation during training
    validate_interval: int = 1000                                   # Interval (in steps) to run validation
    num_validation_episodes: int = 100                              # Number of episodes to validate on
    test_save_dir: Path = Path("runs/validation_results")           # Directory to save validation results

@dataclass
class Test(BaseStrategyConfig):
    """子类， Configuration for test-only mode."""
    test_config_id: str = "test"                               # Unique identifier
    is_test: bool = True                                       # Enable testing mode
    test_save_dir: Path = Path("runs/test_results")            # Directory to save test results
    test_data_length: int = -1                               # Number of samples to test on

# Register test configurations， unqiue确保所有成员的数值是唯一的
@unique
class TestConfigRegistry(Enum):
    PURE_TRAIN = PureTrain
    TRAIN_WITH_VALIDATION = TrainWithValidation  
    TEST = Test

    @property
    def config_id(self) -> str:
        return self.value.test_config_id
    
# Register in ChoiceRegistry。遍历enum类，订阅所有的子类，输入的时候要指定test_config_id
for test_variant in TestConfigRegistry:
    BaseStrategyConfig.register_subclass(test_variant.config_id, test_variant.value)

@dataclass
class RunConfig:
    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under                      # Directory to save test results

    # Test/Validate configuration
    test: BaseStrategyConfig = field(
        default_factory=BaseStrategyConfig.get_choice_class("default")
    )              

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy
        
        # Lift test configuration parameters for backward compatibility
        self.is_test = self.test.is_test
        self.is_validate = self.test.is_validate
        self.test_save_dir = self.test.test_save_dir





