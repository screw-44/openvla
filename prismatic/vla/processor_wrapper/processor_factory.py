"""
processor_factory.py

Factory function to create preprocessor and postprocessor pipelines.
Does NOT depend on lerobot's ProcessorPipeline - implements a simple custom version.
"""

from typing import Dict, Any, List, Optional, Callable
import torch

from prismatic.overwatch import initialize_overwatch
from .processor_steps import (
    TrajectoryRetrievalProcessorStep,
    TrajectoryCompressionProcessorStep,
    VLATokenizerProcessorStep,
    VLAActionDecoderProcessorStep,
)

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


class SimplePipeline:
    """
    Simple sequential pipeline for chaining processor steps.
    
    Does NOT depend on lerobot library - custom lightweight implementation.
    Each step is a callable that takes a dict and returns a dict.
    """
    
    def __init__(self, steps: List[Callable], name: str = "pipeline"):
        """
        Args:
            steps: List of processor step instances
            name: Pipeline name for logging
        """
        self.steps = steps
        self.name = name
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all steps sequentially.
        
        Args:
            data: Input dictionary
        
        Returns:
            Transformed dictionary after all steps
        """
        for i, step in enumerate(self.steps):
            try:
                data = step(data)
            except Exception as e:
                overwatch.error(
                    f"Error in {self.name} step {i} ({step.__class__.__name__}): {e}"
                )
                raise
        
        return data
    
    def __repr__(self) -> str:
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"SimplePipeline(name={self.name}, steps={step_names})"


def make_openvla_processors(
    vla_model,
    config_dict: Dict[str, Any],
    dataset_ref: Optional[Any] = None,
):
    """
    Create preprocessor and postprocessor pipelines for OpenVLA.
    
    Args:
        vla_model: Loaded VLA model instance
        config_dict: Configuration dictionary from config.json
        dataset_ref: Optional reference to dataset (for trajectory retrieval)
    
    Returns:
        Tuple of (preprocessor, postprocessor) SimplePipeline instances
    """
    overwatch.info("Creating OpenVLA processor pipelines...")
    
    # Extract configuration
    vla_config = config_dict.get('vla', {})
    dataset_config = config_dict.get('dataset', {})
    
    # Get trajectory converter from the VLA model instance
    # (ensures consistency with the model's loaded converter)
    trajectory_converter = vla_model.trajectory_converter
    
    # === Build Preprocessor Pipeline ===
    preprocess_steps = []
    
    # Step 1: Trajectory Retrieval (if dataset reference provided)
    if dataset_ref is not None:
        overwatch.info("Adding TrajectoryRetrievalProcessorStep")
        preprocess_steps.append(
            TrajectoryRetrievalProcessorStep(
                dataset_ref=dataset_ref,
                exp_type=dataset_config.get('exp_type', 'positional'),
            )
        )
    
    # Step 2: Trajectory Compression (if dataset reference provided)
    if dataset_ref is not None:
        overwatch.info(
            f"Adding TrajectoryCompressionProcessorStep "
            f"(method={dataset_config.get('trajectory_compression', 'bining')})"
        )
        
        # Get compression kwargs if specified
        compression_kwargs = {}
        if 'compression_kwargs' in dataset_config:
            compression_kwargs = dataset_config['compression_kwargs']
        
        preprocess_steps.append(
            TrajectoryCompressionProcessorStep(
                compression_method=dataset_config.get('trajectory_compression', 'bining'),
                **compression_kwargs,
            )
        )
    
    # Step 3: VLA Tokenization
    # Note: For inference, we may not need the full tokenizer step
    # since predict_action handles this internally
    # But we keep it here for training compatibility
    
    # === Build Postprocessor Pipeline ===
    postprocess_steps = []
    
    # Step 1: Action Decoder
    overwatch.info("Adding VLAActionDecoderProcessorStep")
    postprocess_steps.append(
        VLAActionDecoderProcessorStep(trajectory_converter)
    )
    
    # Create pipelines
    preprocessor = SimplePipeline(preprocess_steps, name="preprocessor")
    postprocessor = SimplePipeline(postprocess_steps, name="postprocessor")
    
    overwatch.info(f"Created preprocessor: {preprocessor}")
    overwatch.info(f"Created postprocessor: {postprocessor}")
    
    return preprocessor, postprocessor
