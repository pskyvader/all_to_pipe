"""
Prompt types and processor for All-to-Pipe.

Handles positive and negative prompt containers and template parsing.
"""

from typing import Optional, Any, List, Tuple, Dict
import torch
import math


class PositivePrompt:
    """
    Strongly-typed container for positive prompt data.
    Attributes are added explicitly as needed.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = (
        "characters",
        "age",
        "body",
        "race",
        "face",
        "hair",
        "clothes",
        "accessories",
        "location",
        "action",
        "pose",
        "camera",
        "lighting",
        "style",
        "color",
        "environment",
        "embeddings",
        "tags",
    )

    def __init__(self) -> None:
        """Initialize empty positive prompt container."""
        self.template: Optional[str] = None
        self.allow_missing: bool = True
        self.model: Optional[str] = None
        self.lora: Optional[str] = None


class NegativePrompt:
    """
    Strongly-typed container for negative prompt data.
    Kept separate from PositivePrompt by design.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = (
        "permanent",
        "embeddings",
        "style",
        "color",
        "tags",
    )

    def __init__(self) -> None:
        """Initialize empty negative prompt container."""
        self.template: Optional[str] = None
        self.allow_missing: bool = True
        self.model: Optional[str] = None
        self.lora: Optional[str] = None

    
    
    
    
    



import torch
import math
from typing import List, Any, Tuple, Dict, Union

class PromptProcessor:
    MAX_PROMPT_TOKENS: int = 75 
    START_TOKEN: int = 49406
    END_TOKEN: int = 49407
    DECAY_K: float = 0.0034055 
    DECAY_FLOOR: float = 0.5

    @staticmethod
    def _normalize_tokens(tokenized: Any) -> torch.Tensor:
        if isinstance(tokenized, dict):
            value: Any = tokenized.get("l", next(iter(tokenized.values())))
        else:
            value = tokenized
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                value = value[0]
            return torch.tensor(value, dtype=torch.float32)
        return value.to(torch.float32)

    @staticmethod
    def extract_prompt_tokens(tokens: torch.Tensor) -> torch.Tensor:
        ids: torch.Tensor = tokens[:, 0]
        mask: torch.Tensor = (ids != 0) & (ids != 49406) & (ids != 49407)
        return tokens[mask]

    @staticmethod
    def _finalize_block(block: torch.Tensor) -> Dict[str, List[List[Tuple[int, float]]]]:
        if block.shape[0] < 77:
            padding: torch.Tensor = torch.zeros((77 - block.shape[0], 2), device=block.device)
            block = torch.cat([block, padding], dim=0)
        pairs: List[Tuple[int, float]] = [(int(row[0]), float(row[1])) for row in block[:77]]
        return {"l": [pairs]}

    @classmethod
    def encode_prompt(cls, prompt_text: str, clip: Any) -> List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        if not prompt_text.strip():
            raise ValueError("Prompt text cannot be empty.")

        segments: List[str] = [s.strip() for s in prompt_text.split(",") if s.strip()]
        seg_data: List[Tuple[torch.Tensor, int]] = []
        for seg in segments:
            tokenized: Any = clip.tokenize(seg)
            pairs: torch.Tensor = cls.extract_prompt_tokens(cls._normalize_tokens(tokenized))
            seg_data.append((pairs, int(pairs.shape[0])))

        chunks_pairs: List[torch.Tensor] = []
        current_pairs: List[torch.Tensor] = []
        current_len: int = 0
        for pairs, length in seg_data:
            if current_len + length <= cls.MAX_PROMPT_TOKENS:
                current_pairs.append(pairs)
                current_len += length
            else:
                if current_pairs:
                    chunks_pairs.append(torch.cat(current_pairs, dim=0))
                current_pairs, current_len = [pairs], length
        if current_pairs:
            chunks_pairs.append(torch.cat(current_pairs, dim=0))

        encoded_orig: List[Any] = []
        encoded_decay: List[Any] = []
        global_token_idx: int = 0
        for chunk in chunks_pairs:
            orig_block: torch.Tensor = torch.cat([
                torch.tensor([[cls.START_TOKEN, 1.0]], device=chunk.device),
                chunk,
                torch.tensor([[cls.END_TOKEN, 1.0]], device=chunk.device)
            ], dim=0)
            
            decayed_chunk: torch.Tensor = chunk.clone()
            for j in range(len(decayed_chunk)):
                m: float = cls.DECAY_FLOOR + (1.0 - cls.DECAY_FLOOR) * math.exp(-cls.DECAY_K * global_token_idx)
                decayed_chunk[j, 1] *= m
                global_token_idx += 1
            
            decayed_block: torch.Tensor = torch.cat([
                torch.tensor([[cls.START_TOKEN, 1.0]], device=chunk.device),
                decayed_chunk,
                torch.tensor([[cls.END_TOKEN, 1.0]], device=chunk.device)
            ], dim=0)

            encoded_orig.append(clip.encode_from_tokens_scheduled(cls._finalize_block(orig_block)))
            encoded_decay.append(clip.encode_from_tokens_scheduled(cls._finalize_block(decayed_block)))

        all_conds: List[torch.Tensor] = []
        for c in encoded_orig:
            v: Any = c[0][0]
            all_conds.append(next(iter(v.values())) if isinstance(v, dict) else v)
        full_cond: torch.Tensor = torch.cat(all_conds, dim=1)

        all_pooled: List[torch.Tensor] = []
        for c in encoded_decay:
            p: Any = c[0][1]
            all_pooled.append(p["pooled_output"] if isinstance(p, dict) else p)
        avg_pooled: torch.Tensor = torch.mean(torch.stack(all_pooled), dim=0)
        
        first_p: Any = encoded_orig[0][0][1]
        final_p: Any = {"pooled_output": avg_pooled} if isinstance(first_p, dict) else avg_pooled

        # Standard ComfyUI format: [[cond_tensor, {"pooled_output": ...}]]
        return [[full_cond, final_p]]