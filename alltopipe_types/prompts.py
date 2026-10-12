import torch
import math
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ClipLimits:
    """Everything the chunking pipeline needs from a CLIP, resolved once.

    Keeping this separate from the CLIP object lets the tokenize/decay/describe
    stages run without touching model weights or performing an encode.
    """

    model_type: str
    max_len: int
    chunk_limit: int
    ref_key: str
    start_id: int
    end_id: int
    tokenizer: Any


class PromptProcessor:
    """Handles multi-encoder synchronization, positional decay, and tensor aggregation."""

    # Comma token text. Used as the word separator when merging weight runs, so the
    # report reads like the original prompt instead of a flat token list.
    RUN_SEPARATOR: str = ","

    @staticmethod
    def _count_suffix(count: int) -> str:
        """Marks a run's token count when it covers more than one token."""
        return f"*{count}" if count > 1 else ""

    @classmethod
    def count_summary_tokens(cls, summary: str) -> int:
        """Total tokens in a run summary, including any hidden from the text."""
        total: int = 0
        for part in summary.split("; "):
            _, _, rest = part.lstrip("(").rstrip(")").partition(",")
            _, _, count_text = rest.partition("*")
            total += int(count_text) if count_text else 1
        return total

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
    def resolve_stream_tokenizer(clip: Any, key: str) -> Any:
        """Locates the tokenizer owning an encoder key: 'l'->clip_l, 'g'->clip_g, 't5xxl'->t5xxl."""
        tokenizer: Any = getattr(clip.tokenizer, f"clip_{key}", None)
        if tokenizer is None:
            tokenizer = getattr(clip.tokenizer, key)
        return tokenizer

    @staticmethod
    def get_stream_delimiters(tokenizer: Any) -> tuple[int, int]:
        """Reads a stream's own start/end token IDs from its tokenizer."""
        start_id: int = int(getattr(tokenizer, "start_token", None) or 0)
        end_id: int = int(getattr(tokenizer, "end_token", None) or 0)
        return start_id, end_id

    @staticmethod
    def classify_architecture(keys: list[str]) -> str:
        """Identifies model type from a tokenizer's encoder keys."""
        if "t5xxl" in keys:
            return "t5_hybrid"  # SD3, Flux
        if "g" in keys and "l" in keys:
            return "sdxl"  # SDXL, Pony
        return "sd15"  # SD1.5

    @classmethod
    def get_clip_limits(cls, clip: Any) -> ClipLimits:
        """Resolves everything the chunking pipeline needs, using one tokenizer probe."""
        probe: dict[str, Any] = clip.tokenize("")
        keys: list[str] = list(probe.keys())
        if not keys:
            raise ValueError("CLIP tokenizer returned no encoder streams.")

        ref_key: str = "l" if "l" in keys else keys[0]
        tokenizer: Any = cls.resolve_stream_tokenizer(clip, ref_key)
        start_id, end_id = cls.get_stream_delimiters(tokenizer)
        max_len: int = int(getattr(tokenizer, "max_length", 77))

        return ClipLimits(
            model_type=cls.classify_architecture(keys),
            max_len=max_len,
            chunk_limit=max_len - 2,
            ref_key=ref_key,
            start_id=start_id,
            end_id=end_id,
            tokenizer=tokenizer,
        )

    @classmethod
    def clean_token_streams(
        cls, clip: Any, token_data: dict[str, list[list[tuple[int, float]]]]
    ) -> dict[str, list[tuple[int, float]]]:
        """Strips each stream's own start/end/pad tokens across every chunk.

        tokenize() already splits long prompts into 77-token chunks and all of
        them are consumed here. Reading only the first chunk silently discarded
        the rest, which on a 200-word prompt dropped 695 of 770 tokens and
        forced the chunk count to 1.

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

    @classmethod
    def tokenize_chunks(
        cls, clip: Any, text: str, limits: ClipLimits
    ) -> list[dict[str, list[tuple[int, float]]]]:
        """Tokenizes a prompt into aligned chunks, one entry per chunk index.

        Every encoder is sliced at the same text index so chunk N of 'l', 'g' and
        't5xxl' stay aligned. An encoder with fewer tokens yields an empty list for
        its trailing chunks.
        """
        if not text.strip():
            raise ValueError("Prompt text cannot be empty.")

        clean_streams: dict[str, list[tuple[int, float]]] = cls.clean_token_streams(
            clip, clip.tokenize(text)
        )
        num_tokens: int = len(clean_streams[limits.ref_key])
        num_chunks: int = max(1, math.ceil(num_tokens / limits.chunk_limit))

        chunks: list[dict[str, list[tuple[int, float]]]] = []
        for index in range(num_chunks):
            start_idx: int = index * limits.chunk_limit
            end_idx: int = (index + 1) * limits.chunk_limit
            chunks.append(
                {k: tokens[start_idx:end_idx] for k, tokens in clean_streams.items()}
            )
        return chunks

    @staticmethod
    def get_chunk_multiplier(
        chunk_index: int, decay: float, decay_floor: float
    ) -> float:
        """Weight multiplier for a whole chunk, driven by chunk index.

        Per-chunk rather than per-token so short prompts are never taxed for
        length, and so the multiplier reported for a chunk always matches the
        weights shown inside it. decay=0.0 is an exact no-op, making chunking
        lossless by default.
        """
        return max(1 - decay * chunk_index, decay_floor)

    @staticmethod
    def scale_weights(
        tokens: list[tuple[int, float]], multiplier: float
    ) -> list[tuple[int, float]]:
        """Applies one multiplier to every token in a segment."""
        return [(token, weight * multiplier) for token, weight in tokens]

    @classmethod
    def apply_decay_to_chunks(
        cls,
        chunks: list[dict[str, list[tuple[int, float]]]],
        decay: float,
        decay_floor: float,
    ) -> list[dict[str, list[tuple[int, float]]]]:
        """Applies a single falloff multiplier to each chunk, preserving alignment.

        Pure: no clip needed, since a chunk's multiplier follows from its index.
        """
        decayed_chunks: list[dict[str, list[tuple[int, float]]]] = []
        for index, chunk in enumerate(chunks):
            multiplier: float = cls.get_chunk_multiplier(index, decay, decay_floor)
            decayed_chunks.append(
                {k: cls.scale_weights(tokens, multiplier) for k, tokens in chunk.items()}
            )
        return decayed_chunks

    @staticmethod
    def decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
        """Decodes token IDs one at a time so each stays aligned with its own weight.

        Batch decoding is avoided here: it merges text across tokens and drops the
        1:1 index alignment with the weights that the pair list needs.
        """
        inv_vocab: dict[int, str] = getattr(tokenizer, "inv_vocab", {})
        decode: Any = getattr(tokenizer, "decode", None)
        if decode is None:
            return [str(inv_vocab.get(t, "")) for t in token_ids]
        return [decode([t]) for t in token_ids]

    @classmethod
    def group_runs(
        cls, pairs: list[tuple[str, float]], tolerance: float = 1e-4
    ) -> list[tuple[str, float, int]]:
        """Collapses sequential runs of equal weight into (text, weight, count).

        Only consecutive tokens are merged, so a weight that reappears later
        after a different weight in between starts a new run. That preserves the
        reading order, which is what makes the output scannable.

        A comma token contributes no word of its own: it acts as the separator
        between the words around it, so the rendered text mirrors the prompt the
        model actually sees. Every token is still counted, including commas and
        any that end a run, so counts stay truthful against the encoder input.
        """
        runs: list[list[Any]] = []
        pending_separator: bool = False
        leading_separators: int = 0

        for text, weight in pairs:
            is_separator: bool = text.strip() == cls.RUN_SEPARATOR
            rounded: float = round(weight, 4)

            if is_separator:
                # Defer: the comma only renders if a word follows it, but it is
                # always counted so totals stay truthful.
                pending_separator = True
                if runs:
                    runs[-1][2] += 1
                else:
                    leading_separators += 1
                continue

            if runs and abs(runs[-1][1] - rounded) <= tolerance:
                joiner: str = ", " if pending_separator else " "
                runs[-1][0] += joiner + text
                runs[-1][2] += 1
            else:
                runs.append([text, rounded, 1])
                # Leading commas belong to the run that follows them.
                if leading_separators:
                    runs[-1][2] += leading_separators
                    leading_separators = 0

            pending_separator = False

        if not runs and leading_separators:
            # Nothing but commas: report them so the token count is not lost.
            return [(cls.RUN_SEPARATOR, cls.get_chunk_multiplier(0, 0.0, 0.0), leading_separators)]

        return [(str(text), w, int(count)) for text, w, count in runs]

    @classmethod
    def format_runs(cls, runs: list[tuple[str, float, int]]) -> str:
        """Renders weight runs as '(one two three:1.0); (four:0.5)'."""
        return "; ".join(
            f"({text}:{weight:g}{cls._count_suffix(count)})"
            for text, weight, count in runs
        )

    @classmethod
    def describe_chunks(
        cls,
        chunks: list[dict[str, list[tuple[int, float]]]],
        limits: ClipLimits,
        chunk_limit: int,
        decay: float,
        decay_floor: float,
    ) -> list[tuple[int, float, str]]:
        """Renders decayed chunks as (start_position, multiplier, run_summary).

        The summary merges consecutive tokens that share a weight, which keeps a
        long prompt readable: with per-chunk decay almost every weight equals the
        chunk multiplier, so most of the output collapses to one run.

        Only the reference encoder is decoded; the others exist to keep alignment
        for encoding. The reported multiplier is the same one applied to the
        weights, so the two never disagree.

        Comma tokens are kept and used to separate words inside a run, so the
        report mirrors the prompt the model actually sees.
        """
        described: list[tuple[int, float, str]] = []
        for index, chunk in enumerate(chunks):
            position: int = index * chunk_limit
            tokens: list[tuple[int, float]] = chunk.get(limits.ref_key, [])
            texts: list[str] = cls.decode_tokens(
                limits.tokenizer, [token for token, _ in tokens]
            )
            pairs: list[tuple[str, float]] = list(
                zip(texts, (w for _, w in tokens))
            )
            multiplier: float = cls.get_chunk_multiplier(index, decay, decay_floor)
            described.append((position, multiplier, cls.format_runs(cls.group_runs(pairs))))
        return described

    @staticmethod
    def warn_if_over_budget(num_chunks: int, limits: ClipLimits) -> None:
        """Reports prompts long enough to degrade prompt adherence."""
        budget: int = PromptProcessor.get_chunk_budget(limits.model_type)
        if num_chunks > budget:
            logging.warning(
                "Prompt produced %d chunks (%d tokens), above the %d-chunk budget "
                "for %s. Prompt adherence may degrade.",
                num_chunks,
                num_chunks * limits.chunk_limit,
                budget,
                limits.model_type,
            )

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
        cls, clip: Any, chunks: list[dict[str, list[tuple[int, float]]]], limits: ClipLimits
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Pads and encodes each chunk, returning one cond tensor and pooled output per chunk.

        Every block handed to CLIP is exactly limits.max_len tokens, since CLIP's
        positional embedding table has a fixed width.
        """
        cls.warn_if_over_budget(len(chunks), limits)

        cond_list: list[torch.Tensor] = []
        pooled_list: list[torch.Tensor] = []

        for chunk in chunks:
            formatted_block: dict[str, Any] = cls.wrap_and_pad_block(
                chunk, limits.start_id, limits.end_id, limits.max_len
            )
            encoded: list[Any] = clip.encode_from_tokens_scheduled(formatted_block)
            cond_list.append(encoded[0][0])
            pooled_list.append(cls.extract_pooled_output(encoded))

        return cond_list, pooled_list

    @classmethod
    def build_conditioning(
        cls,
        cond_list: list[torch.Tensor],
        pooled_list: list[torch.Tensor],
        limits: ClipLimits,
        width: int,
        height: int,
        crop_w: int,
        crop_h: int,
        target_width: int,
        target_height: int,
    ) -> list[list[torch.Tensor | dict[str, Any]]]:
        """Concatenates chunk tensors into ComfyUI CONDITIONING and injects metadata."""
        full_cond: torch.Tensor = torch.cat(cond_list, dim=1)

        # Pooled output from the first chunk represents the primary context
        metadata: dict[str, Any] = {"pooled_output": pooled_list[0]}

        # Micro-conditioning for SDXL architecture
        if limits.model_type == "sdxl":
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

        return [[full_cond, metadata]]

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
    ) -> list[list[torch.Tensor | dict[str, Any]]]:
        """
        Main entry point. Synchronizes multi-encoder tokens and aggregates tensors.
        Explicitly raises ValueError on empty strings to prevent downstream sampler errors.
        """
        limits: ClipLimits = cls.get_clip_limits(clip)
        chunks: list[dict[str, list[tuple[int, float]]]] = cls.apply_decay_to_chunks(
            cls.tokenize_chunks(clip, text, limits),
            decay,
            decay_floor,
        )
        cond_list, pooled_list = cls.encode_chunks(clip, chunks, limits)
        return cls.build_conditioning(
            cond_list,
            pooled_list,
            limits,
            width,
            height,
            crop_w,
            crop_h,
            target_width,
            target_height,
        )

    @classmethod
    def describe_prompt(
        cls, clip: Any, text: str, decay: float, decay_floor: float
    ) -> list[tuple[int, float, str]]:
        """Returns (position, multiplier, token pairs) per chunk without encoding.

        Runs no forward pass and loads no weights, so callers can inspect chunking
        on a loaded clip for free.
        """
        limits: ClipLimits = cls.get_clip_limits(clip)
        chunks: list[dict[str, list[tuple[int, float]]]] = cls.apply_decay_to_chunks(
            cls.tokenize_chunks(clip, text, limits),
            decay,
            decay_floor,
        )
        cls.warn_if_over_budget(len(chunks), limits)
        return cls.describe_chunks(
            chunks, limits, limits.chunk_limit, decay, decay_floor
        )
