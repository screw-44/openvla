"""
processor_wrapper

Processor wrapper for OpenVLA to make it compatible with LeRobot evaluation framework.
All code in this module wraps existing VLA implementation without modifying it.
"""

from .policy_wrapper import OpenVLAPolicyWrapper
from .processor_steps import (
    TrajectoryRetrievalProcessorStep,
    TrajectoryCompressionProcessorStep,
    VLATokenizerProcessorStep,
    VLAActionDecoderProcessorStep,
)
from .processor_factory import make_openvla_processors, SimplePipeline

__all__ = [
    "OpenVLAPolicyWrapper",
    "TrajectoryRetrievalProcessorStep",
    "TrajectoryCompressionProcessorStep",
    "VLATokenizerProcessorStep",
    "VLAActionDecoderProcessorStep",
    "make_openvla_processors",
    "SimplePipeline",
]
