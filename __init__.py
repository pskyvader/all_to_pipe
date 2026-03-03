"""
All-to-Pipe module for ComfyUI.

A custom node module to build, manipulate, and export reusable generation pipelines
containing models, LoRAs, parameters, and prompt structures.
"""

from typing import Dict, Type, Any

from .nodes.pipe_node import PipeNode
from .nodes.positive_prompt_node import PositivePromptNode
from .nodes.negative_prompt_node import NegativePromptNode
from .nodes.model_node import ModelNode
from .nodes.lora_node import LoraNode
from .nodes.parameters_builder_node import ParametersBuilderNode
from .nodes.image_config_node import ImageConfigNode
from .nodes.template_node import TemplateNode
from .nodes.export_node import ExportNode
from .nodes.export_json_node import ExportJsonNode

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS: Dict[str, Type[Any]] = {
    "AllToPipe_Create": PipeNode,
    "AllToPipe_PositivePrompt": PositivePromptNode,
    "AllToPipe_NegativePrompt": NegativePromptNode,
    "AllToPipe_Model": ModelNode,
    "AllToPipe_LoRA": LoraNode,
    "AllToPipe_Parameters": ParametersBuilderNode,
    "AllToPipe_ImageConfig": ImageConfigNode,
    "AllToPipe_Template": TemplateNode,
    "AllToPipe_SamplerExport": ExportNode,
    "AllToPipe_JsonExport": ExportJsonNode,
}

# Human-readable display name mappings
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "AllToPipe_Create": "All-to-Pipe: Create Pipe",
    "AllToPipe_PositivePrompt": "All-to-Pipe: Positive Prompt",
    "AllToPipe_NegativePrompt": "All-to-Pipe: Negative Prompt",
    "AllToPipe_Model": "All-to-Pipe: Model",
    "AllToPipe_LoRA": "All-to-Pipe: LoRA",
    "AllToPipe_Parameters": "All-to-Pipe: Parameters",
    "AllToPipe_ImageConfig": "All-to-Pipe: Image Config",
    "AllToPipe_Template": "All-to-Pipe: Template",
    "AllToPipe_SamplerExport": "All-to-Pipe: Export (Sampler)",
    "AllToPipe_JsonExport": "All-to-Pipe: Export (JSON)",
}

# CRITICAL: Register custom types used by All-to-Pipe nodes
# This is required for ComfyUI to understand and validate node connections
CUSTOM_TYPE_NAMES = [
    "PIPE",
]


