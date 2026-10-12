"""
Unit tests for the shared constants.

The validators and the nodes both read these ranges, so a range that is empty,
inverted, or missing its bound would silently reject every value.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from custom_nodes.all_to_pipe.common import constants


RANGES: tuple[tuple[str, int | float, int | float], ...] = (
    ("MIN_STEPS", constants.MIN_STEPS, constants.MAX_STEPS),
    ("MIN_CFG", constants.MIN_CFG, constants.MAX_CFG),
    ("MIN_SEED", constants.MIN_SEED, constants.MAX_SEED),
    ("MIN_LORA_WEIGHT", constants.MIN_LORA_WEIGHT, constants.MAX_LORA_WEIGHT),
    ("MIN_CLIP_WEIGHT", constants.MIN_CLIP_WEIGHT, constants.MAX_CLIP_WEIGHT),
    ("MIN_CLIP_SKIP", constants.MIN_CLIP_SKIP, constants.MAX_CLIP_SKIP),
    ("MIN_WIDTH", constants.MIN_WIDTH, constants.MAX_WIDTH),
    ("MIN_HEIGHT", constants.MIN_HEIGHT, constants.MAX_HEIGHT),
    ("MIN_BATCH_SIZE", constants.MIN_BATCH_SIZE, constants.MAX_BATCH_SIZE),
    ("MIN_NOISE", constants.MIN_NOISE, constants.MAX_NOISE),
    ("MIN_DECAY", constants.MIN_DECAY, constants.MAX_DECAY),
    ("MIN_DENOISE", constants.MIN_DENOISE, constants.MAX_DENOISE),
)


class TestRanges:
    @pytest.mark.parametrize("name,minimum,maximum", RANGES)
    def test_bounds_are_ordered(self, name: str, minimum: float, maximum: float) -> None:
        assert minimum < maximum, name

    def test_fractional_fields_span_zero_to_one(self) -> None:
        for minimum, maximum in (
            (constants.MIN_CFG, constants.MAX_CFG),
            (constants.MIN_NOISE, constants.MAX_NOISE),
            (constants.MIN_DECAY, constants.MAX_DECAY),
            (constants.MIN_DENOISE, constants.MAX_DENOISE),
        ):
            assert minimum <= 0.0
            assert maximum >= 1.0

    def test_clip_skip_stays_negative(self) -> None:
        # ModelProcessor.load_model only accepts a negative layer index.
        assert constants.MAX_CLIP_SKIP <= -1


class TestChoices:
    def test_samplers_are_available(self) -> None:
        assert "euler" in constants.SUPPORTED_SAMPLERS

    def test_schedulers_are_available(self) -> None:
        assert "normal" in constants.SUPPORTED_SCHEDULERS