import os
from typing import List, Tuple
import folder_paths
import comfy.utils
import comfy.model_patcher
import comfy.sd


class LoraSpec:
    def __init__(
        self,
        name: str,
        subfolder: str,
        weight: float,
        clip_weight: float,
    ) -> None:
        self.name: str = name
        self.subfolder: str = subfolder
        self.weight: float = weight
        self.clip_weight: float = clip_weight


class LoraProcessor:
    @staticmethod
    def apply_lora(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP,
        loras: List[LoraSpec],
    ) -> Tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        if not loras:
            return (model, clip)

        patched_model = model
        patched_clip = clip

        for lora in loras:
            target_path = os.path.join(lora.subfolder, lora.name)
            lora_path = folder_paths.get_full_path("loras", target_path)

            if not lora_path:
                raise FileNotFoundError(f"LoRA '{target_path}' not found.")

            lora_weights = comfy.utils.load_torch_file(lora_path)

            patched_model, patched_clip = comfy.sd.load_lora_for_models(
                patched_model, patched_clip, lora_weights, lora.weight, lora.clip_weight
            )

        return (patched_model, patched_clip)
