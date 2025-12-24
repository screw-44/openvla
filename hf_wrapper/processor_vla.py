"""
Processor pipeline for VLA policy following lerobot standards.

Following the ACT policy pattern, this module exports a make_vla_pre_post_processors function
that creates the appropriate PolicyProcessorPipeline objects for VLA policies.
"""

from typing import Any
from dataclasses import dataclass

import torch

from lerobot.processor import (
    DeviceProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.pipeline import PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from lerobot.configs.policies import PreTrainedConfig


@dataclass
@ProcessorStepRegistry.register(name="vla_image_processor")
class VLAImageProcessorStep(ObservationProcessorStep):
    """什么都不做的步骤"""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_vla_pre_post_processors(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """最小化的处理器 - 图像处理放在模型里"""

    input_steps = [
        # 只做基本操作，不处理图像
        RenameObservationsProcessorStep(rename_map={}),
        DeviceProcessorStep(device="cuda"),
    ]

    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
