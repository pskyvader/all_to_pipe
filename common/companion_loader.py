"""
Companion file loader for All-to-Pipe.

Loads and manages JSON companion files for models and LoRAs.
"""

from typing import Optional, Dict, Any, List, Union
import json
import os
from dataclasses import dataclass, field


@dataclass
class CompanionFile:
    """
    Represents a companion JSON file for a model or LoRA.

    Provides metadata, limits, and prompt suggestions.
    """

    sampler: List[str] = field(default_factory=list)
    """List of samplers. Single value means select one from list."""

    steps: List[Union[int, List[int]]] = field(default_factory=list)
    """
    Steps configuration:
    - len=1: use that specific value
    - len=2: range [min, max], select random in range
    - len>2: choose one from list
    """

    resolution: List[List[int]] = field(default_factory=list)
    """List of valid [width, height] pairs."""

    cfg: List[Union[float, List[float]]] = field(default_factory=list)
    """Cfg configuration (same as steps)."""

    negative_prompt: List[str] = field(default_factory=list)
    """List of negative prompts to randomly select from."""

    positive_prompt: List[str] = field(default_factory=list)
    """List of positive prompts to randomly select from."""

    raw_data: Dict[str, Any] = field(default_factory=dict)
    """Raw dictionary data from JSON file."""


class CompanionLoader:
    """
    Loader for companion JSON files.

    Handles discovering, loading, and parsing companion files
    for models and LoRAs.
    """

    @staticmethod
    def load_model_companion(
        model_name: str,
        subfolder: str,
        base_path: str = "models/checkpoints",
    ) -> Optional[CompanionFile]:
        """
        Load companion JSON file for a model.

        Looks for {model_name}.json alongside the model file.

        Args:
            model_name: Model filename without extension
            subfolder: Subfolder within models/checkpoints
            base_path: Base path to models directory

        Returns:
            CompanionFile instance if found, None otherwise
        """
        return CompanionLoader._load_companion_file(
            model_name, subfolder, base_path
        )

    @staticmethod
    def load_lora_companion(
        lora_name: str,
        subfolder: str,
        base_path: str = "models/loras",
    ) -> Optional[CompanionFile]:
        """
        Load companion JSON file for a LoRA.

        Looks for {lora_name}.json alongside the LoRA file.

        Args:
            lora_name: LoRA filename without extension
            subfolder: Subfolder within models/loras
            base_path: Base path to loras directory

        Returns:
            CompanionFile instance if found, None otherwise
        """
        return CompanionLoader._load_companion_file(
            lora_name, subfolder, base_path
        )

#TODO: if no companion file is found, print a message as a warning.
    @staticmethod
    def _load_companion_file(
        name: str,
        subfolder: str,
        base_path: str,
    ) -> Optional[CompanionFile]:
        """
        Internal method to load companion JSON file.

        Args:
            name: Base name (without extension)
            subfolder: Subfolder path
            base_path: Base directory path

        Returns:
            CompanionFile instance if found, None otherwise
        """
        if not name:
            return None

        # Build path: base_path/subfolder/name.json
        if subfolder:
            json_path: str = os.path.join(base_path, subfolder, f"{name}.json")
        else:
            json_path = os.path.join(base_path, f"{name}.json")

        # Check if file exists
        if not os.path.isfile(json_path):
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            # Parse JSON into CompanionFile
            return CompanionLoader._parse_companion_data(data)

        except (json.JSONDecodeError, IOError, OSError) as e:
            # Log error silently, return None
            return None


#TODO: group by data type, since they have different ways to modify parameters 
# steps,cfg ar numeric
# width and height are size
# prompts are text

# numeric types:
# if single value, assign it directly 
# if 2 values: check that the current value is within those limits. if not, choose a random number between those limits
# if more values: treat them as a list and check that the current value is exactly equal to one of the values in the list. if not, choose one element randomly

# size types:
# those should come in pairs: [[w,h],[w,h]...]
# if instead they come in string format ("wxh or w,h etc"), first transform to actual pairs of numbers
# check that the width and height correspond exactly to one of the pairs.
# if not, choose a pair randomly from the list

# text types:
# could be a list of terms or a string
# if its a string, first convert to list of terms splitting by comma
# from the list of strings, first shuffle,then choose a random amount of terms to add
# then concatenate with comma
# finally assign to the proper prompt, example:
# if input is:lora file / negative text, assign to negative prompts class, feature "lora"


# always try to find values for all parameters, sizes, lora weight, and maybe even
# things like clip skip, upscale (this should be a sub dictionary and inside follow the main rules)
# detailer, and any other element you find in the file
# the main types should be the ones described before, and a dict type where you process the inside values
# also, remember that if theres any value not handled by this class, print a warning about a not found value


    @staticmethod
    def _parse_companion_data(data: Dict[str, Any]) -> CompanionFile:
        """
        Parse raw JSON data into CompanionFile.

        Args:
            data: Dictionary from JSON file

        Returns:
            Parsed CompanionFile instance
        """
        companion: CompanionFile = CompanionFile()
        companion.raw_data = data

        # Parse sampler
        if "sampler" in data:
            samplers: Any = data["sampler"]
            companion.sampler = samplers if isinstance(samplers, list) else [samplers]

        # Parse steps
        if "steps" in data:
            steps_data: Any = data["steps"]
            companion.steps = steps_data if isinstance(steps_data, list) else [steps_data]

        # Parse resolution
        if "resolution" in data:
            res_data: Any = data["resolution"]
            companion.resolution = res_data if isinstance(res_data, list) else [res_data]

        # Parse cfg
        if "cfg" in data:
            cfg_data: Any = data["cfg"]
            companion.cfg = cfg_data if isinstance(cfg_data, list) else [cfg_data]

        # Parse negative_prompt
        if "negative_prompt" in data:
            neg_data: Any = data["negative_prompt"]
            companion.negative_prompt = (
                neg_data if isinstance(neg_data, list) else [neg_data]
            )

        # Parse positive_prompt
        if "positive_prompt" in data:
            pos_data: Any = data["positive_prompt"]
            companion.positive_prompt = (
                pos_data if isinstance(pos_data, list) else [pos_data]
            )

        return companion
