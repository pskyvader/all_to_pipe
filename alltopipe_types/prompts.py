import torch
import math
import logging
from typing import Any


class PromptContainer:
    """Base container for managing prompt state and metadata."""

    def __init__(self) -> None:
        self.allow_missing: bool = True
        self.model: str | None = None
        self.lora: str | None = None


class PositivePrompt(PromptContainer):
    """Specific container for subject-focused positive features."""

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
    """Specific container for exclusion-focused negative features."""

    ALLOWED_FEATURES: tuple[str, ...] = (
        "permanent",
        "embeddings",
        "style",
        "color",
        "tags",
    )


class PromptProcessor:
    """Handles multi-encoder synchronization, positional decay, and tensor aggregation."""

    DECAY_K: float = 0.0
    DECAY_FLOOR: float = 0.5

    # Chunk counts past which prompt adherence is expected to degrade. CLIP-L and
    # CLIP-G were trained at 77 tokens, so SD15/SDXL lose coherence well before
    # T5's 512-token limit.
    CHUNK_BUDGETS: dict[str, int] = {
        "sd15": 4,  # 308 tokens, ~4x the CLIP training length
        "sdxl": 4,  # 308 tokens
        "t5_hybrid": 6,  # 462 tokens, just under T5's 512
    }

    @classmethod
    def get_chunk_budget(cls, model_type: str) -> int:
        """Maximum chunks before prompt adherence is expected to degrade."""
        return cls.CHUNK_BUDGETS.get(model_type, 4)

    @staticmethod
    def detect_architecture(clip: Any) -> str:
        """Identifies model type to determine metadata requirements."""
        test_tokens: dict[str, Any] = clip.tokenize("")
        keys: list[str] = list(test_tokens.keys())

        if "t5xxl" in keys:
            return "t5_hybrid"  # SD3, Flux
        if "g" in keys and "l" in keys:
            return "sdxl"  # SDXL, Pony
        return "sd15"  # SD1.5

    @staticmethod
    def get_tokenizer_limits(clip: Any) -> tuple[int, int]:
        """Retrieves max sequence length and internal chunking limits."""
        tokenizer: Any = (
            clip.tokenizer.clip_l
            if hasattr(clip.tokenizer, "clip_l")
            else clip.tokenizer
        )
        max_len: int = getattr(tokenizer, "max_length", 77)
        return int(max_len), int(max_len - 2)

    @staticmethod
    def resolve_stream_tokenizer(clip: Any, key: str) -> Any:
        """Locates the tokenizer owning an encoder key: 'l'->clip_l, 'g'->clip_g, 't5xxl'->t5xxl."""
        tokenizer: Any = getattr(clip.tokenizer, f"clip_{key}", None)
        if tokenizer is None:
            tokenizer = getattr(clip.tokenizer, key)
        return tokenizer

    @staticmethod
    def get_stream_delimiters(clip: Any, key: str) -> tuple[int, int]:
        """Reads a stream's own start/end token IDs from its tokenizer."""
        tokenizer: Any = PromptProcessor.resolve_stream_tokenizer(clip, key)
        start_id: int = int(getattr(tokenizer, "start_token", None) or 0)
        end_id: int = int(getattr(tokenizer, "end_token", None) or 0)
        return start_id, end_id

    @classmethod
    def clean_token_streams(
        cls, clip: Any, token_data: dict[str, list[list[tuple[int, float]]]]
    ) -> dict[str, list[tuple[int, float]]]:
        """Strips each stream's own start/end/pad tokens across every chunk.

        tokenize() already splits long prompts into 77-token chunks and all of
        them are consumed here. Reading only the first chunk silently discarded
        the rest, which on a 200-word prompt dropped 695 of 770 tokens and
        forced num_chunks to 1.

        Per-encoder because CLIP pads 'l' with its end token (49407) while 'g'
        pads with 0, so a single shared delimiter set leaves stray EOTs behind.
        """
        streams: dict[str, list[tuple[int, float]]] = {}
        for k, chunks in token_data.items():
            tokenizer: Any = cls.resolve_stream_tokenizer(clip, k)
            removable: set[int] = {
                int(token)
                for token in (
                    getattr(tokenizer, "start_token", None),
                    getattr(tokenizer, "end_token", None),
                    getattr(tokenizer, "pad_token", None),
                )
                if token is not None
            }
            tokens: list[tuple[int, float]] = []
            for chunk in chunks:
                tokens.extend(
                    (int(t), w) for t, w in chunk if int(t) not in removable
                )
            streams[k] = tokens
        return streams

    @staticmethod
    def decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
        """Decodes token IDs one at a time so each stays aligned with its own weight.

        Batch decoding is avoided here: it merges text across tokens and drops the
        1:1 index alignment with chunk_content weights that the pair list needs.
        """
        inv_vocab: dict[int, str] = getattr(tokenizer, "inv_vocab", {})
        decode: Any = getattr(tokenizer, "decode", None)
        if decode is None:
            return [str(inv_vocab.get(t, "")) for t in token_ids]
        return [decode([t]) for t in token_ids]

    @classmethod
    def apply_decay_to_segment(
        cls, segment: list[tuple[int, float]], start_offset: int
    ) -> list[tuple[int, float]]:
        """Calculates and applies the exponential weight decay to a chunk segment."""
        decayed: list[tuple[int, float]] = []

        for i, (token_id, weight) in enumerate(segment):
            global_pos: int = start_offset + i
            # multiplier: float = cls.DECAY_FLOOR + (1.0 - cls.DECAY_FLOOR) * math.exp(
            #     -cls.DECAY_K * global_pos
            # )
            multiplier: float = max(1 - cls.DECAY_K * global_pos, cls.DECAY_FLOOR)

            decayed.append((token_id, weight * multiplier))
        return decayed

    @staticmethod
    def wrap_and_pad_block(
        streams: dict[str, list[tuple[int, float]]],
        start_id: int,
        end_id: int,
        max_len: int,
    ) -> dict[str, list[list[tuple[int, float]]]]:
        """Re-inserts delimiters and pads streams to the required CLIP sequence length."""
        block: dict[str, list[list[tuple[int, float]]]] = {}
        for k, tokens in streams.items():
            formatted: list[tuple[int, float]] = (
                [(start_id, 1.0)] + tokens + [(end_id, 1.0)]
            )
            if len(formatted) < max_len:
                formatted += [(0, 0.0)] * (max_len - len(formatted))
            block[k] = [formatted[:max_len]]
        return block

    @staticmethod
    def extract_pooled_output(encoded_result: list[Any]) -> torch.Tensor:
        """Safely extracts the pooled_output tensor from the CLIP encoding result."""
        # result format: [[tensor, {"pooled_output": tensor}]]
        data: dict[str, Any] | torch.Tensor = encoded_result[0][1]
        if isinstance(data, dict) and "pooled_output" in data:
            return data["pooled_output"]
        return data  # Fallback for SD1.5/Simple encoders

    @classmethod
    def encode_chunks(
        cls,
        clip: Any,
        token_data: dict[str, list[list[tuple[int, float]]]],
        start_id: int,
        end_id: int,
        max_len: int,
        chunk_limit: int,
        max_chunks: int,
        model_type: str,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[int, float, str]],
    ]:
        """Slices every encoder in lockstep, applies decay, encodes, records chunk text.

        Returns per-chunk (start_position, decay_multiplier, "[(token_text, weight), ...]").
        Chunk boundaries are recoverable as position // chunk_limit.
        """
        clean_streams: dict[str, list[tuple[int, float]]] = cls.clean_token_streams(
            clip, token_data
        )
        ref_key: str = "l" if "l" in clean_streams else next(iter(clean_streams))
        text_tokenizer: Any = cls.resolve_stream_tokenizer(clip, ref_key)

        num_tokens: int = len(clean_streams[ref_key])
        num_chunks: int = max(1, math.ceil(num_tokens / chunk_limit))

        if num_chunks > max_chunks:
            logging.warning(
                "Prompt produced %d chunks (%d tokens), above the %d-chunk budget "
                "for %s. Prompt adherence may degrade.",
                num_chunks,
                num_chunks * chunk_limit,
                max_chunks,
                model_type,
            )

        cond_list: list[torch.Tensor] = []
        pooled_list: list[torch.Tensor] = []
        chunked_text: list[tuple[int, float, str]] = []

        for i in range(num_chunks):
            start_idx: int = i * chunk_limit
            end_idx: int = (i + 1) * chunk_limit
            chunk_content: dict[str, list[tuple[int, float]]] = {}

            for k in clean_streams:
                # Ensure all encoders (G, L, T5) are sliced at the exact same text index
                segment: list[tuple[int, float]] = clean_streams[k][start_idx:end_idx]
                chunk_content[k] = cls.apply_decay_to_segment(segment, start_idx)

            # Format and Encode
            formatted_block: dict[str, Any] = cls.wrap_and_pad_block(
                chunk_content, start_id, end_id, max_len
            )
            encoded: list[Any] = clip.encode_from_tokens_scheduled(formatted_block)

            cond_list.append(encoded[0][0])
            pooled_list.append(cls.extract_pooled_output(encoded))

            ref_ids: list[int] = [t for t, _ in chunk_content[ref_key]]
            ref_texts: list[str] = cls.decode_tokens(text_tokenizer, ref_ids)
            pairs: list[tuple[str, float]] = [
                (text, round(weight, 4))
                for text, (_tid, weight) in zip(ref_texts, chunk_content[ref_key])
            ]
            multiplier: float = max(1 - cls.DECAY_K * start_idx, cls.DECAY_FLOOR)
            chunked_text.append((start_idx, multiplier, str(pairs)))

        return cond_list, pooled_list, chunked_text

    @classmethod
    def encode_prompt(
        cls,
        clip: Any,
        text: str,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        crop_w: int,
        crop_h: int,
        decay: float,
        decay_floor: float,
    ) -> tuple[
        list[list[torch.Tensor | dict[str, Any]]],
        list[tuple[int, float, str]],
    ]:
        """
        Main entry point. Synchronizes multi-encoder tokens and aggregates tensors.
        Explicitly raises ValueError on empty strings to prevent downstream sampler errors.
        """
        if not text.strip():
            raise ValueError("Prompt text cannot be empty.")

        cls.DECAY_K = decay
        cls.DECAY_FLOOR = decay_floor

        # 1. Initialization and Limit Detection
        model_type: str = cls.detect_architecture(clip)
        max_len, chunk_limit = cls.get_tokenizer_limits(clip)

        # 2. Tokenization
        token_data: dict[str, list[list[tuple[int, float]]]] = clip.tokenize(text)

        # 3. Chunked Encoding and Text Capture
        ref_key: str = "l" if "l" in token_data else next(iter(token_data))
        start_id, end_id = cls.get_stream_delimiters(clip, ref_key)

        cond_list, pooled_list, chunked_text = cls.encode_chunks(
            clip,
            token_data,
            start_id,
            end_id,
            max_len,
            chunk_limit,
            cls.get_chunk_budget(model_type),
            model_type,
        )

        # 4. Final Aggregation and Metadata Injection
        full_cond: torch.Tensor = torch.cat(cond_list, dim=1)

        # Pooled output from the first chunk represents the primary context
        metadata: dict[str, Any] = {"pooled_output": pooled_list[0]}

        # Micro-conditioning for SDXL architecture
        if model_type == "sdxl":
            metadata.update(
                {
                    "width": width,
                    "height": height,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "target_width": target_width,
                    "target_height": target_height,
                }
            )

        return ([[full_cond, metadata]], chunked_text)
