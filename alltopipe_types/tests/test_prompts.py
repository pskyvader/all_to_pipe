"""
Unit tests for prompt chunking, delimiter stripping, and positional decay.

Uses stub tokenizers and a stub clip so no model weights are loaded.
"""

import math
import sys
import logging
import pytest
import torch
from typing import Any
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types.prompts import PromptProcessor

BOS: int = 49406
EOS: int = 49407
PAD: int = 0
MAX_LEN: int = 77

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


def content_tokens(count: int) -> list[tuple[int, float]]:
    """Builds content tokens that cannot collide with PAD/BOS/EOS delimiters."""
    return [(FIRST_CONTENT_ID + i, 1.0) for i in range(count)]


class StubTokenizers:
    """Mimics the attribute layout of SDXLTokenizer / SD3Tokenizer."""

    def __init__(self) -> None:
        self.clip_l = StubTokenizer(pad_token=EOS)
        self.clip_g = StubTokenizer(pad_token=PAD)


class StubClip:
    """Exposes only the surface PromptProcessor uses."""

    def __init__(
        self,
        streams: dict[str, list[list[tuple[int, float]]]],
        use_inv_vocab_only: bool = False,
    ) -> None:
        self.token_data = streams
        self.tokenizer = StubTokenizers()
        self.encoded_lengths: list[int] = []
        self.encoded_keys: list[list[str]] = []
        self.use_inv_vocab_only = use_inv_vocab_only

    def tokenize(self, text: str) -> dict[str, list[list[tuple[int, float]]]]:
        return self.token_data

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, Any]
    ) -> list[Any]:
        # Records what the encoder was handed, so padding/length can be asserted.
        self.encoded_lengths.append(len(tokens["l"][0]))
        self.encoded_keys.append(sorted(tokens.keys()))
        width = len(tokens["l"][0])
        return [[torch.zeros(1, width, 8), {"pooled_output": torch.zeros(1, 8)}]]


@pytest.fixture(autouse=True)
def reset_decay_defaults() -> Any:
    """Keeps tests that mutate class-level decay from leaking into each other."""
    k: float = PromptProcessor.DECAY_K
    floor: float = PromptProcessor.DECAY_FLOOR
    yield
    PromptProcessor.DECAY_K = k
    PromptProcessor.DECAY_FLOOR = floor


def make_streams(
    content: list[tuple[int, float]], key: str = "l"
) -> dict[str, list[list[tuple[int, float]]]]:
    """Wraps content tokens in BOS/content/EOS then pads to max_length like comfy."""
    start: int = BOS
    end: int = EOS
    pad: int = EOS if key == "l" else PAD
    body: list[tuple[int, float]] = [(start, 1.0)] + content + [(end, 1.0)]
    body += [(pad, 1.0)] * (MAX_LEN - len(body))
    return {key: [body]}


def make_t5_streams(
    content: list[tuple[int, float]]
) -> dict[str, list[list[tuple[int, float]]]]:
    """t5xxl variant: no start token, end token 1, padded with 0."""
    body: list[tuple[int, float]] = list(content) + [(1, 1.0)]
    body += [(PAD, 1.0)] * (MAX_LEN - len(body))
    return {"t5xxl": [body]}


def make_sdxl_streams(content: list[tuple[int, float]]) -> dict[str, list[list[tuple[int, float]]]]:
    """Two-encoder stream set, as SDXL tokenizes."""
    streams: dict[str, list[list[tuple[int, float]]]] = make_streams(content, key="l")
    streams.update(make_streams(content, key="g"))
    return streams


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


def count_pairs(pairs_str: str) -> int:
    """Counts top-level entries in a str()'d list of (text, weight) tuples."""
    return pairs_str.count("), ")


class TestCleanTokenStreams:
    def test_strips_repeated_eos_from_l_stream(self) -> None:
        """Regression: EOS 49407 padding must not survive as prompt content.

        The previous implementation derived delimiters from the first stream key
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
        """t5xxl reports start_token=None, which must not collapse to 0."""
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
        pinned num_chunks to 1, which made the decay feature unreachable.
        """
        num_chunks = 10
        clip = StubClip(make_multi_chunk_streams(content_tokens(75), num_chunks))
        cleaned = PromptProcessor.clean_token_streams(clip, clip.token_data)

        assert len(cleaned["l"]) == 75 * num_chunks

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


class TestDecayDefaults:
    def test_decay_default_is_lossless(self) -> None:
        """0.0 must make every multiplier exactly 1.0.

        At the previous 0.3 default everything past position 2 was blended halfway
        to a neutral embedding, weakening most of the prompt.
        """
        assert PromptProcessor.DECAY_K == 0.0

    def test_zero_decay_leaves_weights_untouched(self) -> None:
        PromptProcessor.DECAY_K = 0.0
        PromptProcessor.DECAY_FLOOR = 0.5
        segment: list[tuple[int, float]] = content_tokens(50)
        decayed = PromptProcessor.apply_decay_to_segment(segment, 0)

        assert [w for _, w in decayed] == [1.0] * 50


class TestChunkBudget:
    def test_budget_per_architecture(self) -> None:
        assert PromptProcessor.get_chunk_budget("sd15") == 4
        assert PromptProcessor.get_chunk_budget("sdxl") == 4
        assert PromptProcessor.get_chunk_budget("t5_hybrid") == 6

    def test_unknown_architecture_falls_back(self) -> None:
        assert PromptProcessor.get_chunk_budget("something_new") == 4

    def test_warns_when_over_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        clip = StubClip(make_streams(content_tokens(200)))
        with caplog.at_level(logging.WARNING):
            PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 2, "sdxl"
            )
        assert "above the 2-chunk budget" in caplog.text
        assert "sdxl" in caplog.text

    def test_no_warning_at_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        clip = StubClip(make_streams(content_tokens(100)))
        with caplog.at_level(logging.WARNING):
            _c, _p, text = PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 2, "sdxl"
            )
        assert len(text) == 2
        assert "budget" not in caplog.text


class TestDecodeTokens:
    def test_decodes_each_token_individually(self) -> None:
        tokenizer: StubTokenizer = StubTokenizer(inv_vocab={320: "a", 736: "red"})
        assert PromptProcessor.decode_tokens(tokenizer, [320, 736]) == ["a", "red"]

    def test_falls_back_to_inv_vocab_without_decode(self) -> None:
        class NoDecode:
            inv_vocab: dict[int, str] = {320: "a", 736: "red"}

        assert PromptProcessor.decode_tokens(NoDecode(), [320, 736]) == ["a", "red"]

    def test_falls_back_to_empty_string_for_unknown_id(self) -> None:
        class NoDecode:
            inv_vocab: dict[int, str] = {}

        assert PromptProcessor.decode_tokens(NoDecode(), [320]) == [""]


class TestEncodeChunks:
    def test_short_prompt_is_single_chunk(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        cond, pooled, text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert len(cond) == 1
        assert len(pooled) == 1
        assert len(text) == 1

    def test_no_spurious_chunk_from_padding(self) -> None:
        """A 17-token prompt must stay 1 chunk instead of the buggy 2."""
        clip = StubClip(make_streams(content_tokens(17)))
        cond, _pooled, text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert len(cond) == 1
        assert len(text) == 1

    def test_long_prompt_chunks_at_limit(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        cond, _pooled, _text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert len(cond) == math.ceil(160 / (MAX_LEN - 2))

    def test_positions_continue_across_chunks(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        _cond, _pooled, text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert [entry[0] for entry in text] == [0, 75, 150]

    def test_multiplier_matches_decay_at_position(self) -> None:
        PromptProcessor.DECAY_K = 0.1
        PromptProcessor.DECAY_FLOOR = 0.5
        try:
            clip = StubClip(make_streams(content_tokens(160)))
            _cond, _pooled, text = PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
            )
            for position, multiplier, _pairs in text:
                assert multiplier == max(1 - 0.1 * position, 0.5)
        finally:
            PromptProcessor.DECAY_K = 0.3
            PromptProcessor.DECAY_FLOOR = 0.5

    def test_multiplier_floors_at_decay_floor(self) -> None:
        PromptProcessor.DECAY_K = 0.1
        PromptProcessor.DECAY_FLOOR = 0.5
        try:
            clip = StubClip(make_streams(content_tokens(400)))
            _cond, _pooled, text = PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
            )
            assert text[-1][1] == 0.5
        finally:
            PromptProcessor.DECAY_K = 0.3
            PromptProcessor.DECAY_FLOOR = 0.5

    def test_pair_count_matches_chunk_content_length(self) -> None:
        total = 160
        clip = StubClip(make_streams(content_tokens(total)))
        _cond, _pooled, text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        limit = MAX_LEN - 2
        for position, _multiplier, pairs in text:
            expected = min(limit, total - position)
            assert count_pairs(pairs) == expected - 1

    def test_pair_weights_are_decayed_and_rounded(self) -> None:
        PromptProcessor.DECAY_K = 0.1
        PromptProcessor.DECAY_FLOOR = 0.5
        try:
            clip = StubClip(make_streams([(320, 1.0), (736, 1.0), (2368, 1.0)]))
            _cond, _pooled, text = PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
            )
            assert text[0][2] == "[('<320>', 1.0), ('<736>', 0.9), ('<2368>', 0.8)]"
        finally:
            PromptProcessor.DECAY_K = 0.3
            PromptProcessor.DECAY_FLOOR = 0.5

    def test_second_chunk_pair_weights_continue_decay(self) -> None:
        """Chunk 1 starts at position 75, so its weights are already floored."""
        PromptProcessor.DECAY_K = 0.1
        PromptProcessor.DECAY_FLOOR = 0.5
        try:
            clip = StubClip(make_streams(content_tokens(160)))
            _cond, _pooled, text = PromptProcessor.encode_chunks(
                clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
            )
            assert text[1][0] == 75
            assert text[1][2].startswith("[('<1075>', 0.5), ")
        finally:
            PromptProcessor.DECAY_K = 0.3
            PromptProcessor.DECAY_FLOOR = 0.5

    def test_pair_text_contains_no_special_tokens(self) -> None:
        clip = StubClip(make_streams(content_tokens(20)))
        _cond, _pooled, text = PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert "''," not in text[0][2]
        assert "('" in text[0][2]

    def test_every_encoded_block_is_exactly_max_len(self) -> None:
        """CLIP's positional table is [77, 768]; any other length raises."""
        clip = StubClip(make_streams(content_tokens(160)))
        PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert clip.encoded_lengths == [MAX_LEN] * 3

    def test_multi_encoder_block_contains_all_keys(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(10)))
        PromptProcessor.encode_chunks(
            clip, clip.token_data, BOS, EOS, MAX_LEN, MAX_LEN - 2, 99, "sd15"
        )

        assert clip.encoded_keys == [["g", "l"]]


class TestEncodePrompt:
    def test_returns_conditioning_and_chunk_text(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        cond, text = PromptProcessor.encode_prompt(
            clip,
            "a red cat",
            1024,
            1024,
            1024,
            1024,
            crop_w=0,
            crop_h=0,
            decay=0.1,
            decay_floor=0.5,
        )

        assert isinstance(cond, list) and isinstance(cond[0], list)
        assert len(cond[0]) == 2
        assert isinstance(text, list) and isinstance(text[0], tuple)

    def test_cond_tensor_spans_all_chunks(self) -> None:
        clip = StubClip(make_streams(content_tokens(160)))
        cond, text = PromptProcessor.encode_prompt(
            clip,
            "long prompt",
            1024,
            1024,
            1024,
            1024,
            crop_w=0,
            crop_h=0,
            decay=0.1,
            decay_floor=0.5,
        )

        assert cond[0][0].shape[1] == MAX_LEN * len(text)

    def test_sd15_omits_micro_conditioning(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        cond, _text = PromptProcessor.encode_prompt(
            clip,
            "a red cat",
            1024,
            768,
            1024,
            768,
            crop_w=3,
            crop_h=5,
            decay=0.1,
            decay_floor=0.5,
        )

        assert "width" not in cond[0][1]
        assert "crop_w" not in cond[0][1]

    def test_sdxl_injects_micro_conditioning(self) -> None:
        clip = StubClip(make_sdxl_streams(content_tokens(10)))
        cond, _text = PromptProcessor.encode_prompt(
            clip,
            "a red cat",
            1024,
            768,
            1024,
            768,
            crop_w=3,
            crop_h=5,
            decay=0.1,
            decay_floor=0.5,
        )

        assert cond[0][1]["width"] == 1024
        assert cond[0][1]["height"] == 768
        assert cond[0][1]["crop_w"] == 3
        assert cond[0][1]["crop_h"] == 5
        assert cond[0][1]["target_width"] == 1024
        assert cond[0][1]["target_height"] == 768

    def test_pooled_output_always_present(self) -> None:
        clip = StubClip(make_streams(content_tokens(10)))
        cond, _text = PromptProcessor.encode_prompt(
            clip,
            "a red cat",
            1024,
            1024,
            1024,
            1024,
            crop_w=0,
            crop_h=0,
            decay=0.1,
            decay_floor=0.5,
        )

        assert "pooled_output" in cond[0][1]

    def test_empty_text_raises(self) -> None:
        clip = StubClip(make_streams([(320, 1.0)]))
        with pytest.raises(ValueError):
            PromptProcessor.encode_prompt(
                clip,
                "   ",
                1024,
                1024,
                1024,
                1024,
                crop_w=0,
                crop_h=0,
                decay=0.1,
                decay_floor=0.5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
