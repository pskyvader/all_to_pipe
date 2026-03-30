import os
from typing import List, Tuple, Optional, Any, Dict, Set
import folder_paths
import comfy.utils
import comfy.model_patcher
import comfy.sd
import torch
import logging

logger = logging.getLogger("AllToPipe")
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
        self.cached_lora: Optional[Any] = None

class LoraProcessor:
    @staticmethod
    def load_lora(lora: LoraSpec) -> Dict[str, torch.Tensor]:
        if not lora.cached_lora:
            target_path = os.path.join(lora.subfolder, lora.name)
            lora_path = folder_paths.get_full_path("loras", target_path)

            if not lora_path:
                raise FileNotFoundError(f"LoRA '{target_path}' not found.")

            lora.cached_lora:Dict[str, torch.Tensor]= comfy.utils.load_torch_file(lora_path)
        return lora.cached_lora

    @staticmethod
    def get_model_key_set(model: comfy.model_patcher.ModelPatcher) -> Set[str]:
        return set(model.model.state_dict().keys())

    @staticmethod
    def is_lora_compatible(lora_weights: Dict[str, torch.Tensor], model_keys: Set[str]) -> bool:
        if not lora_weights:
            return False

        sample_keys: List[str] = list(lora_weights.keys())[:50]
        matches: int = 0
        
        for l_key in sample_keys:
            clean_key: str = l_key.replace("model.diffusion_model.", "diffusion_model.")
            clean_key = clean_key.replace("lora_unet_", "diffusion_model.")
            
            if any(clean_key in m_key for m_key in model_keys):
                matches += 1
        
        return (matches / len(sample_keys)) >= 0.10

    @staticmethod
    def apply_lora(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP,
        loras: List[Any],
    ) -> Tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        if not loras:
            return (model, clip)

        model_keys: Set[str] = LoraProcessor.get_model_key_set(model)
        patched_model = model
        patched_clip = clip

        for lora in loras:
            try:
                lora_weights: Dict[str, torch.Tensor] = LoraProcessor.load_lora(lora)
                
                if not LoraProcessor.is_lora_compatible(lora_weights, model_keys):
                    logger.warning(f"Architecture Mismatch: Skipping {lora.name}")
                    continue

                patched_model, patched_clip = comfy.sd.load_bypass_lora_for_models(
                    patched_model,
                    patched_clip,
                    lora_weights,
                    lora.weight,
                    lora.clip_weight,
                )
            except Exception as e:
                logger.error(f"Failed to load {lora.name}: {e}")
                continue

        return (patched_model, patched_clip)