import os
from typing import Tuple
import folder_paths
import comfy.sd
import comfy.model_patcher

class Model:
    def __init__(self, name: str, subfolder: str = "") -> None:
        self.name: str = name
        self.subfolder: str = subfolder

class ModelProcessor:
    @staticmethod
    def load_model(model: Model) -> Tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP, comfy.sd.VAE]:
        if not model or not model.name:
            raise ValueError("Model name is required and cannot be empty")
        
        target_path = os.path.join(model.subfolder, model.name)
        ckpt_path = folder_paths.get_full_path("checkpoints", target_path)
        
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint '{target_path}' not found.")

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )

        return (out[0], out[1], out[2])
