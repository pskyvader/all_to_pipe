from typing import Optional, Any, List, Tuple, Dict, Union
import torch
import math


class PromptContainer:
    """Base container for prompt data."""

    def __init__(self) -> None:
        self.template: Optional[str] = None
        self.allow_missing: bool = True
        self.model: Optional[str] = None
        self.lora: Optional[str] = None


class PositivePrompt(PromptContainer):
    """Strongly-typed container for positive prompt features."""

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


class NegativePrompt(PromptContainer):
    """Strongly-typed container for negative prompt features."""

    ALLOWED_FEATURES: tuple[str, ...] = (
        "permanent",
        "embeddings",
        "style",
        "color",
        "tags",
    )


class PromptProcessor:
    """
    Handles synchronized multi-encoder tokenization,
    positional decay, and global tensor aggregation.
    """

    DECAY_K: float = 0.0034055
    DECAY_FLOOR: float = 0.5

    @classmethod
    def encode_prompt(
        cls, prompt_text: str, clip: Any
    ) -> List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        if not prompt_text.strip():
            raise ValueError("Prompt text cannot be empty.")

        # 1. Environment Detection
        token_data: Dict[str, List[List[Tuple[int, float]]]] = clip.tokenize(
            prompt_text
        )
        main_key: str = "l" if "l" in token_data else next(iter(token_data))

        # Access nested tokenizer if present (common in multi-encoder CLIP models)
        tokenizer = (
            clip.tokenizer.clip_l
            if hasattr(clip.tokenizer, "clip_l")
            else clip.tokenizer
        )
        max_len: int = getattr(tokenizer, "max_length", 77)
        chunk_limit: int = max_len - 2

        # Extract special tokens from the primary encoder's sample
        sample_tokens: List[Tuple[int, float]] = token_data[main_key][0]
        start_id: int = sample_tokens[0][0]
        end_id: int = sample_tokens[-1][0]

        # 2. Extract Raw Content Streams
        clean_streams: Dict[str, List[Tuple[int, float]]] = {}
        for k, weight_list in token_data.items():
            clean_streams[k] = [
                t for t in weight_list[0] if t[0] not in (start_id, end_id, 0)
            ]

        # 3. Synchronized Chunking & Positional Decay
        num_tokens: int = len(clean_streams[main_key])
        num_chunks: int = math.ceil(num_tokens / chunk_limit) if num_tokens > 0 else 1

        processed_chunks: List[Dict[str, List[List[Tuple[int, float]]]]] = []

        for i in range(num_chunks):
            start_idx: int = i * chunk_limit
            end_idx: int = start_idx + chunk_limit
            chunk_structure: Dict[str, List[List[Tuple[int, float]]]] = {}

            for k, stream in clean_streams.items():
                segment: List[Tuple[int, float]] = stream[start_idx:end_idx]

                # Apply decay strictly to the primary encoder's weights
                if k == main_key:
                    decayed: List[Tuple[int, float]] = []
                    for j, (tid, weight) in enumerate(segment):
                        global_pos: int = start_idx + j
                        # Decayed weight: weight * m
                        m: float = cls.DECAY_FLOOR + (1.0 - cls.DECAY_FLOOR) * math.exp(
                            -cls.DECAY_K * global_pos
                        )
                        decayed.append((tid, weight * m))
                    segment = decayed

                # Wrap segment with start/end tokens and pad to max_len
                block: List[Tuple[int, float]] = (
                    [(start_id, 1.0)] + segment + [(end_id, 1.0)]
                )
                if len(block) < max_len:
                    block += [(0, 0.0)] * (max_len - len(block))

                chunk_structure[k] = [block[:max_len]]

            processed_chunks.append(chunk_structure)

        # 4. Encoding and Aggregation
        cond_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []

        for chunk in processed_chunks:
            encoded_data: List[Any] = clip.encode_from_tokens_scheduled(chunk)

            # Extract sequence conditioning tensor
            c_tensor: torch.Tensor = encoded_data[0][0]
            cond_list.append(c_tensor)

            # Extract pooled output (handles Dict or raw Tensor)
            p_data: Any = encoded_data[0][1]
            pooled_list.append(
                p_data["pooled_output"] if isinstance(p_data, dict) else p_data
            )

        # 5. Final Assembly
        # Concatenate sequence tokens and average global pooled vectors
        full_cond: torch.Tensor = torch.cat(cond_list, dim=1)
        avg_pooled: torch.Tensor = torch.mean(torch.stack(pooled_list), dim=0)

        return [[full_cond, {"pooled_output": avg_pooled}]]
