"""
Unit tests for prompt chunking, delimiter stripping, and positional decay.

Uses stub tokenizers and a stub clip so no model weights are loaded.
"""

import math
import sys
import json
import logging
import pytest
import torch
from typing import Any
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types.prompts import ClipLimits, PromptProcessor

BOS: int = 49406
EOS: int = 49407
PAD: int = 0
MAX_LEN: int = 77
CHUNK_LIMIT: int = MAX_LEN - 2

# Token IDs here must avoid BOS/EOS/PAD, since those are stripped as delimiters.
FIRST_CONTENT_ID: int = 1000


class StubTokenizer:
    """Minimal stand-in for comfy SDTokenizer."""

    def __init__(
        self,
        start_token: int | None = BOS,
        end_token: int | None = EOS,
        pad_token: int | None = PAD,
        max_length: int = MAX_LEN,
        inv_vocab: dict[int, str] | None = None,
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.inv_vocab: dict[int, str] = inv_vocab or {}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        # Mirrors CLIP byte-level decode: a single-token decode strips leading space.
        specials: dict[int, str] = {BOS: "", EOS: "", PAD: ""}
        return "".join(specials.get(t, self.inv_vocab.get(t, f"<{t}>")) for t in token_ids)


class StubTokenizers:
    """Mimics the attribute layout of SDXLTokenizer / SD3Tokenizer."""

    def __init__(self) -> None:
        self.clip_l = StubTokenizer(pad_token=EOS)
        self.clip_g = StubTokenizer(pad_token=PAD)


class StubClip:
    """Exposes only the surface PromptProcessor uses.

    encode_from_tokens_scheduled raises unless allow_encode is set, so tests can
    assert that the describe path never performs an encode.
    """

    def __init__(
        self,
        streams: dict[str, list[list[tuple[int, float]]]],
        allow_encode: bool = True,
    ) -> None:
        self.token_data = streams
        self.tokenizer = StubTokenizers()
        self.encoded_lengths: list[int] = []
        self.encoded_keys: list[list[str]] = []
        self.tokenize_calls: int = 0
        self.allow_encode: bool = allow_encode

    def tokenize(self, text: str) -> dict[str, list[list[tuple[int, float]]]]:
        self.tokenize_calls += 1
        return self.token_data

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, Any]
    ) -> list[Any]:
        if not self.allow_encode:
            raise AssertionError("encode called on a describe-only path")
        # Records what the encoder was handed, so padding/length can be asserted.
        self.encoded_lengths.append(len(tokens["l"][0]))
        self.encoded_keys.append(sorted(tokens.keys()))
        width = len(tokens["l"][0])
        return [[torch.zeros(1, width, 8), {"pooled_output": torch.zeros(1, 8)}]]


def content_tokens(count: int) -> list[tuple[int, float]]:
    """Builds content tokens that cannot collide with PAD/BOS/EOS delimiters."""
    return [(FIRST_CONTENT_ID + i, 1.0) for i in range(count)]


def make_streams(
    content: list[tuple[int, float]], key: str = "l"
) -> dict[str, list[list[tuple[int, float]]]]:
    """Wraps content tokens in BOS/content/EOS then pads to max_length like comfy."""
    body: list[tuple[int, float]] = [(BOS, 1.0)] + content + [(EOS, 1.0)]
    pad: int = EOS if key == "l" else PAD
    body += [(pad, 1.0)] * (MAX_LEN - len(body))
    return {key: [body]}


def make_t5_streams(
    content: list[tuple[int, float]]
) -> dict[str, list[list[tuple[int, float]]]]:
    """t5xxl variant: no start token, end token 1, padded with 0."""
    body: list[tuple[int, float]] = list(content) + [(1, 1.0)]
    body += [(PAD, 1.0)] * (MAX_LEN - len(body))
    return {"t5xxl": [body]}


def make_multi_chunk_streams(
    per_chunk: list[tuple[int, float]], num_chunks: int, key: str = "l"
) -> dict[str, list[list[tuple[int, float]]]]:
    """Mimics comfy tokenize() splitting a long prompt into several 77-token chunks.

    Each chunk is padded to max_length independently, exactly as SDTokenizer does,
    so only the delimiters distinguish padding from content. Token IDs are offset
    per chunk so that reordering is detectable.
    """
    chunks: list[list[tuple[int, float]]] = []
    for index in range(num_chunks):
        offset: int = index * len(per_chunk)
        content: list[tuple[int, float]] = [
            (token + offset, weight) for token, weight in per_chunk
        ]
        body: list[tuple[int, float]] = [(BOS, 1.0)] + content + [(EOS, 1.0)]
        pad: int = EOS if key == "l" else PAD
        body += [(pad, 1.0)] * (MAX_LEN - len(body))
        chunks.append(body)
    return {key: chunks}


def make_sdxl_streams(
    content: list[tuple[int, float]]
) -> dict[str, list[list[tuple[int, float]]]]:
    """Two-encoder stream set, as SDXL tokenizes."""
    streams: dict[str, list[list[tuple[int, float]]]] = make_streams(content, key="l")
    streams.update(make_streams(content, key="g"))
    return streams


def stub_limits(clip: StubClip, model_type: str = "sd15") -> ClipLimits:
    """Builds ClipLimits directly, bypassing the tokenizer probe."""
    tokenizer: Any = clip.tokenizer.clip_l
    start_id, end_id = PromptProcessor.get_stream_delimiters(tokenizer)
    return ClipLimits(
        model_type=model_type,
        max_len=MAX_LEN,
        chunk_limit=CHUNK_LIMIT,
        ref_key="l",
        start_id=start_id,
        end_id=end_id,
        tokenizer=tokenizer,
    )


def summary_runs(summary: str) -> list[tuple[str, float, int]]:
    """Parses '(a b:1*2); (c:0.5)' back into (text, weight, count) tuples."""
    runs: list[tuple[str, float, int]] = []
    for part in summary.split("; "):
        body = part.lstrip("(").rstrip(")")
        text, _, rest = body.rpartition(":")
        weight_text, _, count_text = rest.partition("*")
        runs.append((text, float(weight_text), int(count_text) if count_text else 1))
    return runs


def summary_weight(summary: str) -> list[float]:
    """All distinct weights in a summary, in order."""
    return [weight for _text, weight, _count in summary_runs(summary)]


def run_counts(summary: str) -> list[int]:
    """Token count of each run in a summary."""
    return [count for _text, _weight, count in summary_runs(summary)]


class TestGetClipLimits:
    def test_sd15_fields(self) -> None:
        clip = StubClip(make_streams(content_tokens(5)))
        limits = PromptProcessor.get_clip_limits(clip)

        assert limits.model_type == "sd15"
        assert limits.ref_key == "l"
        assert limits.max_len == MAX_LEN
        assert limits.chunk_limit == CHUNK_LIMIT
        assert limits.start_id == BOS
        assert limits.end_id == EOS
        assert limits.tokenizer is clip.tokenizer.clip_l

    def test_sdxl_fields(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(5)))
        limits = PromptProcessor.get_clip_limits(clip)

        assert limits.model_type == "sdxl"
        assert limits.ref_key == "l"

    def test_t5_hybrid_fields(self) -> None:
        clip = StubClip(make_t5_streams([(3, 1.0)]))
        clip.tokenizer.t5xxl = StubTokenizer(
            start_token=None, end_token=1, pad_token=PAD
        )
        limits = PromptProcessor.get_clip_limits(clip)

        assert limits.model_type == "t5_hybrid"
        assert limits.ref_key == "t5xxl"
        # t5xxl has no start token, so it must not collapse to 0
        assert limits.start_id == 0
        assert limits.end_id == 1

    def test_uses_a_single_tokenizer_probe(self) -> None:
        clip = StubClip(make_streams(content_tokens(5)))
        PromptProcessor.get_clip_limits(clip)

        assert clip.tokenize_calls == 1

    def test_empty_streams_raises(self) -> None:
        clip = StubClip({})
        with pytest.raises(ValueError):
            PromptProcessor.get_clip_limits(clip)


class TestCleanTokenStreams:
    def test_strips_repeated_eos_from_l_stream(self) -> None:
        """Regression: EOS 49407 padding must not survive as prompt content.

        The original implementation derived delimiters from the first stream key
        ('g', padded with 0) so 49407 was never stripped, leaving 73 stray EOTs.
        """
        clip = StubClip(make_streams([(320, 1.0), (736, 1.0), (2368, 1.0)]))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["l"]) == 3
        assert all(token != EOS for token, _ in cleaned["l"])

    def test_strips_pad_from_g_stream(self) -> None:
        clip = StubClip(make_streams([(320, 1.0), (736, 1.0)], key="g"))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["g"]) == 2
        assert all(token != PAD for token, _ in cleaned["g"])

    def test_tolerates_none_start_token(self) -> None:
        """t5xxl reports start_token=None, which must not become a stripped 0."""
        clip = StubClip(make_t5_streams([(3, 1.0), (9, 1.0)]))
        clip.tokenizer.t5xxl = StubTokenizer(
            start_token=None, end_token=1, pad_token=PAD
        )
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["t5xxl"]) == 2
        assert [token for token, _ in cleaned["t5xxl"]] == [3, 9]

    def test_multi_encoder_streams_aligned(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(10)))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["l"]) == len(cleaned["g"]) == 10

    def test_consumes_every_chunk_not_just_the_first(self) -> None:
        """Regression: tokenize() splits long prompts into chunks.

        Reading only chunk[0] dropped 695 of 770 tokens on a 200-word prompt and
        pinned the chunk count to 1, which made the decay feature unreachable.
        """
        num_chunks = 10
        clip = StubClip(make_multi_chunk_streams(content_tokens(CHUNK_LIMIT), num_chunks))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["l"]) == CHUNK_LIMIT * num_chunks

    def test_chunks_are_consumed_in_order(self) -> None:
        clip = StubClip(make_multi_chunk_streams(content_tokens(3), 4))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        ids = [token for token, _ in cleaned["l"]]
        assert ids == [FIRST_CONTENT_ID + i for i in range(12)]

    def test_delimiters_stripped_from_every_chunk(self) -> None:
        """Each chunk carries its own BOS/EOS, so all must be removed."""
        clip = StubClip(make_multi_chunk_streams(content_tokens(10), 3))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert all(token != BOS for token, _ in cleaned["l"])
        assert all(token != EOS for token, _ in cleaned["l"])
        assert all(token != PAD for token, _ in cleaned["l"])


class TestTokenizeChunks:
    def test_short_prompt_is_single_chunk(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)

        assert len(chunks) == 1
        assert len(chunks[0]["l"]) == 10

    def test_chunks_at_limit(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)

        assert len(chunks) == math.ceil(160 / CHUNK_LIMIT)

    def test_chunk_major_shape(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(10)))
        limits = stub_limits(clip, "sdxl")
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert sorted(chunks[0].keys()) == ["g", "l"]

    def test_no_spurious_chunk_from_padding(self) -> None:
        """A 17-token prompt must stay 1 chunk instead of the buggy 2."""
        clip = StubClip(make_streams(content_tokens(17)))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)

        assert len(chunks) == 1

    def test_encoders_align_across_chunks(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(100)))
        limits = stub_limits(clip, "sdxl")
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)

        for index, chunk in enumerate(chunks):
            expected = min(CHUNK_LIMIT, 100 - index * CHUNK_LIMIT)
            assert len(chunk["l"]) == expected
            assert len(chunk["g"]) == expected

    def test_empty_text_raises(self) -> None:
        clip = StubClip(make_streams(content_tokens(5)))
        limits = stub_limits(clip)

        with pytest.raises(ValueError):
            PromptProcessor.tokenize_chunks(clip, "   ", limits)


class TestGetChunkMultiplier:
    def test_zero_decay_is_lossless(self) -> None:
        """0.0 must make every multiplier exactly 1.0.

        At the previous 0.3 default everything past position 2 was blended halfway
        to a neutral embedding, weakening most of the prompt.
        """
        assert PromptProcessor.get_chunk_multiplier(0, 0.0, 0.5) == 1.0
        assert PromptProcessor.get_chunk_multiplier(7, 0.0, 0.5) == 1.0

    def test_linear_falloff_by_chunk_index(self) -> None:
        assert PromptProcessor.get_chunk_multiplier(0, 0.1, 0.5) == 1.0
        assert PromptProcessor.get_chunk_multiplier(1, 0.1, 0.5) == 0.9
        assert PromptProcessor.get_chunk_multiplier(2, 0.1, 0.5) == 0.8

    def test_respects_floor(self) -> None:
        assert PromptProcessor.get_chunk_multiplier(5, 0.1, 0.5) == 0.5
        assert PromptProcessor.get_chunk_multiplier(99, 0.1, 0.5) == 0.5

    def test_short_prompts_are_never_taxed(self) -> None:
        """Chunk 0 is always full weight regardless of decay strength.

        Per-token decay at 0.1 pinned everything past token 5 to the floor, so a
        short prompt was penalised merely for existing.
        """
        assert PromptProcessor.get_chunk_multiplier(0, 1.0, 0.5) == 1.0


class TestApplyDecay:
    def _chunked(self, total: int, key: str = "l") -> Any:
        clip = StubClip(make_streams(content_tokens(total), key=key))
        limits = stub_limits(clip)
        return PromptProcessor.tokenize_chunks(clip, "text", limits)

    def test_zero_decay_preserves_every_weight(self) -> None:
        chunks = self._chunked(160)
        decayed = PromptProcessor.apply_decay_to_chunks(chunks, 0.0, 0.5)

        for before, after in zip(chunks, decayed):
            assert [w for _, w in before["l"]] == [w for _, w in after["l"]]

    def test_every_token_in_a_chunk_shares_one_multiplier(self) -> None:
        """Decay is per chunk, so a chunk's weights differ only by base weight."""
        chunks = self._chunked(160)
        decayed = PromptProcessor.apply_decay_to_chunks(chunks, 0.1, 0.5)

        assert all(w == 1.0 for _, w in decayed[0]["l"])
        assert all(w == 0.9 for _, w in decayed[1]["l"])
        assert all(w == 0.8 for _, w in decayed[2]["l"])

    def test_respects_floor(self) -> None:
        chunks = self._chunked(600)
        decayed = PromptProcessor.apply_decay_to_chunks(chunks, 0.1, 0.5)

        assert all(w == 0.5 for _, w in decayed[-1]["l"])

    def test_preserves_base_weights(self) -> None:
        """A token already at 1.1 keeps its ratio to a token at 1.0."""
        clip = StubClip(make_streams([(320, 1.1), (736, 1.0)]))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)
        decayed = PromptProcessor.apply_decay_to_chunks(chunks, 0.1, 0.5)

        weights = [round(w, 4) for _, w in decayed[0]["l"]]
        assert weights == [1.1, 1.0]

    def test_preserves_encoder_alignment(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(100)))
        limits = stub_limits(clip, "sdxl")
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)
        decayed = PromptProcessor.apply_decay_to_chunks(chunks, 0.2, 0.5)

        for before, after in zip(chunks, decayed):
            assert sorted(before.keys()) == sorted(after.keys())
            assert len(before["l"]) == len(after["l"])
            assert len(before["g"]) == len(after["g"])

    def test_is_pure(self) -> None:
        """Decay must not mutate the input chunks."""
        chunks = self._chunked(80)
        snapshot = [[list(chunk.values())] for chunk in chunks]

        PromptProcessor.apply_decay_to_chunks(chunks, 0.3, 0.5)

        assert [[list(c.values())] for c in chunks] == snapshot


class TestDecodeTokens:
    def test_decodes_each_token_individually(self) -> None:
        tokenizer = StubTokenizer(inv_vocab={320: "a", 736: "red"})
        assert PromptProcessor.decode_tokens(tokenizer, [320, 736]) == ["a", "red"]

    def test_falls_back_to_inv_vocab_without_decode(self) -> None:
        class NoDecode:
            inv_vocab: dict[int, str] = {320: "a", 736: "red"}

        assert PromptProcessor.decode_tokens(NoDecode(), [320, 736]) == ["a", "red"]

    def test_falls_back_to_empty_string_for_unknown_id(self) -> None:
        class NoDecode:
            inv_vocab: dict[int, str] = {}

        assert PromptProcessor.decode_tokens(NoDecode(), [320]) == [""]


class TestDescribeChunks:
    def _describe(self, clip: StubClip, decay: float = 0.0) -> Any:
        limits = stub_limits(clip)
        chunks = PromptProcessor.apply_decay_to_chunks(
            PromptProcessor.tokenize_chunks(clip, "text", limits),
            decay,
            0.5,
        )
        return PromptProcessor.describe_chunks(chunks, limits, CHUNK_LIMIT, decay, 0.5)

    def test_returns_position_multiplier_and_pairs(self) -> None:
        clip = StubClip(make_streams([(320, 1.0), (736, 1.0), (2368, 1.0)]))
        described = self._describe(clip, decay=0.1)

        # Chunk 0 keeps full weight: a short prompt is never taxed.
        assert described[0][0] == 0
        assert described[0][1] == 1.0
        assert described[0][2] == "(<320> <736> <2368>:1*3)"

    def test_positions_continue_across_chunks(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = self._describe(clip)

        assert [entry[0] for entry in described] == [0, 75, 150]

    def test_multiplier_matches_chunk_index(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = self._describe(clip, decay=0.1)

        assert [entry[1] for entry in described] == [1.0, 0.9, 0.8]

    def test_reported_multiplier_matches_the_weights_inside(self) -> None:
        """The tuple scalar and the runs must never disagree."""
        clip = StubClip(make_streams([(320, 1.0), (736, 1.0)]))
        described = self._describe(clip, decay=0.0)

        for _pos, multiplier, summary in described:
            assert summary_weight(summary) == [multiplier]

    def test_token_counts_sum_to_chunk_length(self) -> None:
        """Run counts must account for every token in the chunk."""
        total = 160
        clip = StubClip(make_streams(content_tokens(total)))
        described = self._describe(clip)

        seen = 0
        for position, _multiplier, summary in described:
            expected = min(CHUNK_LIMIT, total - position)
            assert sum(run_counts(summary)) == expected
            seen += expected
        assert seen == total

    def test_zero_decay_keeps_every_weight_at_one(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = self._describe(clip, decay=0.0)

        for _pos, _multiplier, summary in described:
            assert set(summary_weight(summary)) == {1.0}

    def test_no_special_tokens_in_text(self) -> None:
        clip = StubClip(make_streams(content_tokens(20)))
        described = self._describe(clip)

        assert "''" not in described[0][2]
        assert "<" in described[0][2]

    def test_commas_separate_words_in_a_run(self) -> None:
        """A comma joins the words around it instead of becoming a word."""
        comma = FIRST_CONTENT_ID  # id that the stub decodes to ","
        clip = StubClip(make_streams([(320, 1.0), (comma, 1.0), (736, 1.0)]))
        clip.tokenizer.clip_l.inv_vocab = {comma: ",", 320: "a", 736: "b"}
        described = self._describe(clip)

        assert described[0][2] == "(a, b:1*3)"

    def test_comma_still_counts_toward_the_run(self) -> None:
        """Rendering is cosmetic: the token count must stay truthful."""
        comma = FIRST_CONTENT_ID
        clip = StubClip(make_streams([(320, 1.0), (comma, 1.0), (736, 1.0)]))
        clip.tokenizer.clip_l.inv_vocab = {comma: ",", 320: "a", 736: "b"}
        limits = stub_limits(clip)
        chunks = PromptProcessor.apply_decay_to_chunks(
            PromptProcessor.tokenize_chunks(clip, "text", limits), 0.0, 0.5
        )
        PromptProcessor.encode_chunks(clip, chunks, limits)

        assert len(chunks[0]["l"]) == 3
        assert sum(run_counts(self._describe(clip)[0][2])) == 3

    def test_distinct_weights_form_separate_runs(self) -> None:
        clip = StubClip(make_streams([(320, 1.0), (736, 0.5), (2368, 1.0)]))
        described = self._describe(clip, decay=0.0)

        assert described[0][2] == "(<320>:1); (<736>:0.5); (<2368>:1)"


class TestGroupRuns:
    def test_merges_consecutive_equal_weights(self) -> None:
        pairs = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        assert PromptProcessor.group_runs(pairs) == [("a b c", 1.0, 3)]

    def test_splits_on_weight_change(self) -> None:
        pairs = [("a", 1.0), ("b", 0.5), ("c", 1.0)]
        assert PromptProcessor.group_runs(pairs) == [
            ("a", 1.0, 1),
            ("b", 0.5, 1),
            ("c", 1.0, 1),
        ]

    def test_repeated_weight_after_a_break_starts_a_new_run(self) -> None:
        """Sequential grouping, not a global merge, so reading order survives."""
        pairs = [("a", 1.0), ("b", 0.5), ("c", 1.0), ("d", 1.0)]
        runs = PromptProcessor.group_runs(pairs)
        assert [(t, c) for t, _w, c in runs] == [("a", 1), ("b", 1), ("c d", 2)]

    def test_tolerates_float_noise(self) -> None:
        pairs = [("a", 0.1 + 0.2), ("b", 0.30000000000000004)]
        assert PromptProcessor.group_runs(pairs) == [("a b", 0.3, 2)]

    def test_empty_input(self) -> None:
        assert PromptProcessor.group_runs([]) == []

    def test_format_marks_multi_token_runs(self) -> None:
        runs = [("a b", 1.0, 2), ("c", 0.5, 1)]
        assert PromptProcessor.format_runs(runs) == "(a b:1*2); (c:0.5)"

    def test_format_uses_compact_weight_notation(self) -> None:
        runs = [("a", 0.85, 1)]
        assert PromptProcessor.format_runs(runs) == "(a:0.85)"

    def test_format_uses_colon_not_comma(self) -> None:
        """A comma would be ambiguous now that commas appear inside the text."""
        assert "," not in PromptProcessor.format_runs([("a b", 1.0, 2)])


class TestRunSeparators:
    """Commas act as word separators; every token is still counted."""

    def _count(self, texts: str) -> tuple[list[tuple[str, float, int]], int]:
        """Splits on '|' into tokens; 'c' stands for a comma token."""
        pairs = [("," if t == "c" else t, 1.0) for t in texts.split("|")]
        runs = PromptProcessor.group_runs(pairs)
        return runs, sum(c for _t, _w, c in runs)

    def test_comma_joins_words(self) -> None:
        runs, count = self._count("a|c|b")
        assert runs == [("a, b", 1.0, 3)]
        assert count == 3

    def test_spaces_join_words_without_commas(self) -> None:
        runs, count = self._count("a|b")
        assert runs == [("a b", 1.0, 2)]
        assert count == 2

    def test_leading_comma_is_counted_not_rendered(self) -> None:
        runs, count = self._count("c|a")
        assert runs == [("a", 1.0, 2)]
        assert count == 2

    def test_trailing_comma_is_counted_not_rendered(self) -> None:
        runs, count = self._count("a|c")
        assert runs == [("a", 1.0, 2)]
        assert count == 2

    def test_repeated_commas_collapse_but_are_counted(self) -> None:
        runs, count = self._count("a|c|c|b")
        assert runs == [("a, b", 1.0, 4)]
        assert count == 4

    def test_comma_only_reports_itself(self) -> None:
        runs, count = self._count("c|c|c")
        assert count == 3
        assert sum(c for _t, _w, c in runs) == 3

    @pytest.mark.parametrize(
        "texts",
        ["a|c|b", "c|a", "a|c", "a|c|c|b", "c|c|c", "a|b|c|d", "a|c|b|c|d"],
    )
    def test_token_counts_always_survive(self, texts: str) -> None:
        _runs, count = self._count(texts)
        assert count == len(texts.split("|"))

    def test_comma_at_a_weight_boundary(self) -> None:
        pairs = [("a", 1.0), ("b", 0.5), (",", 0.5), ("c", 0.5)]
        runs = PromptProcessor.group_runs(pairs)
        assert runs == [("a", 1.0, 1), ("b, c", 0.5, 3)]


class TestEncodeChunks:
    def test_every_encoded_block_is_exactly_max_len(self) -> None:
        """CLIP's positional table is [77, 768]; any other length raises."""
        clip = StubClip(make_streams(content_tokens(160)))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)
        PromptProcessor.encode_chunks(clip, chunks, limits)

        assert clip.encoded_lengths == [MAX_LEN, MAX_LEN, MAX_LEN]

    def test_returns_cond_and_pooled_per_chunk(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        limits = stub_limits(clip)
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)
        cond, pooled = PromptProcessor.encode_chunks(clip, chunks, limits)

        assert len(cond) == 3
        assert len(pooled) == 3
        assert cond[0].shape[1] == MAX_LEN

    def test_multi_encoder_block_contains_all_keys(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(10)))
        limits = stub_limits(clip, "sdxl")
        chunks = PromptProcessor.tokenize_chunks(clip, "text", limits)
        PromptProcessor.encode_chunks(clip, chunks, limits)

        assert clip.encoded_keys == [["g", "l"]]


class TestBuildConditioning:
    def test_concatenates_along_sequence_dim(self) -> None:
        limits = stub_limits(StubClip(make_streams(content_tokens(10))))
        cond = [torch.zeros(1, MAX_LEN, 8) for _ in range(3)]
        pooled = [torch.zeros(1, 8)]

        result = PromptProcessor.build_conditioning(
            cond, pooled, limits, 1024, 1024, 0, 0, 1024, 1024
        )

        assert result[0][0].shape[1] == MAX_LEN * 3

    def test_sd15_omits_micro_conditioning(self) -> None:
        limits = stub_limits(StubClip(make_streams(content_tokens(10))), "sd15")
        cond = [torch.zeros(1, MAX_LEN, 8)]
        pooled = [torch.zeros(1, 8)]

        result = PromptProcessor.build_conditioning(
            cond, pooled, limits, 1024, 768, 3, 5, 1024, 768
        )

        assert "width" not in result[0][1]
        assert "crop_w" not in result[0][1]

    def test_sdxl_injects_micro_conditioning(self) -> None:
        limits = stub_limits(StubClip(make_sdxl_streams(content_tokens(10))), "sdxl")
        cond = [torch.zeros(1, MAX_LEN, 8)]
        pooled = [torch.zeros(1, 8)]

        result = PromptProcessor.build_conditioning(
            cond, pooled, limits, 1024, 768, 3, 5, 2048, 768
        )
        meta = result[0][1]

        assert meta["width"] == 1024
        assert meta["height"] == 768
        assert meta["crop_w"] == 3
        assert meta["crop_h"] == 5
        assert meta["target_width"] == 2048
        assert meta["target_height"] == 768

    def test_pooled_output_always_present(self) -> None:
        limits = stub_limits(StubClip(make_streams(content_tokens(10))))
        cond = [torch.zeros(1, MAX_LEN, 8)]
        pooled = [torch.zeros(1, 8)]

        result = PromptProcessor.build_conditioning(
            cond, pooled, limits, 1024, 1024, 0, 0, 1024, 1024
        )

        assert "pooled_output" in result[0][1]


class TestChunkBudget:
    def test_budget_per_architecture(self) -> None:
        assert PromptProcessor.get_chunk_budget("sd15") == 4
        assert PromptProcessor.get_chunk_budget("sdxl") == 4
        assert PromptProcessor.get_chunk_budget("t5_hybrid") == 6

    def test_unknown_architecture_falls_back(self) -> None:
        assert PromptProcessor.get_chunk_budget("something_new") == 4

    def test_warns_when_over_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        clip = StubClip(make_streams(content_tokens(200)))
        limits = stub_limits(clip, "sdxl")
        with caplog.at_level(logging.WARNING):
            PromptProcessor.warn_if_over_budget(6, limits)

        assert "above the 4-chunk budget" in caplog.text
        assert "sdxl" in caplog.text

    def test_no_warning_at_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        clip = StubClip(make_streams(content_tokens(100)))
        limits = stub_limits(clip, "sdxl")
        with caplog.at_level(logging.WARNING):
            PromptProcessor.warn_if_over_budget(4, limits)

        assert "budget" not in caplog.text


class TestEncodePrompt:
    def test_returns_conditioning(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        cond = PromptProcessor.encode_prompt(
            clip, "a red cat", 1024, 1024, 1024, 1024, 0, 0, 0.0, 0.5
        )

        assert isinstance(cond, list) and isinstance(cond[0], list)
        assert len(cond[0]) == 2

    def test_cond_tensor_spans_all_chunks(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        cond = PromptProcessor.encode_prompt(
            clip, "long prompt", 1024, 1024, 1024, 1024, 0, 0, 0.0, 0.5
        )

        assert cond[0][0].shape[1] == MAX_LEN * 3

    def test_empty_text_raises(self) -> None:
        clip = StubClip(make_streams(content_tokens(5)))
        with pytest.raises(ValueError):
            PromptProcessor.encode_prompt(
                clip, "   ", 1024, 1024, 1024, 1024, 0, 0, 0.0, 0.5
            )


class TestDescribePrompt:
    def test_returns_one_entry_per_chunk(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = PromptProcessor.describe_prompt(clip, "text", 0.0, 0.5)

        assert len(described) == math.ceil(160 / CHUNK_LIMIT)
        assert [entry[0] for entry in described] == [0, 75, 150]

    def test_never_encodes(self) -> None:
        """Guards the property TemplateNode relies on to stay cheap.

        describe_prompt only needs the tokenizer; if it ever runs an encode this
        raises instead of silently costing a forward pass.
        """
        clip = StubClip(make_streams(content_tokens(160)), allow_encode=False)
        described = PromptProcessor.describe_prompt(clip, "text", 0.0, 0.5)

        assert len(described) == 3
        assert clip.encoded_lengths == []

    def test_weights_follow_supplied_decay(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = PromptProcessor.describe_prompt(clip, "text", 0.1, 0.5)

        for index, (position, multiplier, _pairs) in enumerate(described):
            assert position == index * CHUNK_LIMIT
            assert multiplier == max(1 - 0.1 * index, 0.5)

    def test_output_is_json_serialisable(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        described = PromptProcessor.describe_prompt(clip, "text", 0.0, 0.5)

        restored = json.loads(json.dumps(described))
        assert restored == [list(entry) for entry in described]

    def test_empty_text_raises(self) -> None:
        clip = StubClip(make_streams(content_tokens(5)))
        with pytest.raises(ValueError):
            PromptProcessor.describe_prompt(clip, "  ", 0.0, 0.5)

    def test_warns_past_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        clip = StubClip(make_streams(content_tokens(600)))
        with caplog.at_level(logging.WARNING):
            PromptProcessor.describe_prompt(clip, "very long", 0.0, 0.5)

        assert "budget" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
