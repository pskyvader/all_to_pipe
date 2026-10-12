"""
Unit tests for the field validators.

Covers what Update Pipe rejects: wrong types, out of range values, and the
fields that only accept one of a fixed set of values.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from custom_nodes.all_to_pipe.alltopipe_types import LoraSpec
from custom_nodes.all_to_pipe.common.validators import (
    validate_bool,
    validate_color_code,
    validate_float,
    validate_image,
    validate_int,
    validate_latent,
    validate_lora_list,
    validate_str,
)


class TestValidateInt:
    def test_returns_the_value(self) -> None:
        assert validate_int(20, "parameters.steps", 1, 100) == 20

    def test_accepts_the_bounds(self) -> None:
        assert validate_int(1, "parameters.steps", 1, 100) == 1
        assert validate_int(100, "parameters.steps", 1, 100) == 100

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            validate_int(500, "parameters.steps", 1, 100)

    def test_rejects_a_float(self) -> None:
        with pytest.raises(ValueError, match="must be an INT"):
            validate_int(20.0, "parameters.steps", 1, 100)

    def test_rejects_a_bool(self) -> None:
        with pytest.raises(ValueError, match="must be an INT"):
            validate_int(True, "parameters.steps", 1, 100)

    def test_rejects_text(self) -> None:
        with pytest.raises(ValueError, match="must be an INT"):
            validate_int("20", "parameters.steps", 1, 100)


class TestValidateFloat:
    def test_returns_a_float(self) -> None:
        assert validate_float(7.5, "parameters.cfg", 0.0, 20.0) == 7.5

    def test_widens_an_int(self) -> None:
        result = validate_float(4, "parameters.cfg", 0.0, 20.0)

        assert result == 4.0
        assert isinstance(result, float)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 20.0"):
            validate_float(50.0, "parameters.cfg", 0.0, 20.0)

    def test_rejects_a_bool(self) -> None:
        with pytest.raises(ValueError, match="must be a FLOAT"):
            validate_float(True, "parameters.cfg", 0.0, 20.0)


class TestValidateBool:
    def test_returns_the_value(self) -> None:
        assert validate_bool(False, "allow_parse_errors") is False

    def test_rejects_an_int(self) -> None:
        with pytest.raises(ValueError, match="must be a BOOLEAN"):
            validate_bool(1, "allow_parse_errors")


class TestValidateStr:
    def test_returns_the_value(self) -> None:
        assert validate_str("euler", "parameters.sampler") == "euler"

    def test_accepts_empty_by_default(self) -> None:
        assert validate_str("", "positive_prompt.age") == ""

    def test_rejects_empty_when_not_allowed(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_str("  ", "parameters.sampler", allow_empty=False)

    def test_rejects_a_value_outside_the_choices(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            validate_str("nope", "parameters.sampler", choices=("euler", "ddim"))

    def test_shortens_a_long_choice_list(self) -> None:
        choices = tuple(str(index) for index in range(20))

        with pytest.raises(ValueError, match=r"in total"):
            validate_str("nope", "parameters.sampler", choices=choices)

    def test_rejects_a_non_string(self) -> None:
        with pytest.raises(ValueError, match="must be a STRING"):
            validate_str(7, "parameters.sampler")


class TestValidateColorCode:
    def test_normalises_to_upper_case(self) -> None:
        assert validate_color_code("#ff0000", "image_config.color_code") == "#FF0000"

    def test_expands_the_short_form(self) -> None:
        assert validate_color_code("f00", "image_config.color_code") == "#FF0000"

    def test_blank_means_random(self) -> None:
        assert validate_color_code("", "image_config.color_code") is None
        assert validate_color_code(None, "image_config.color_code") is None

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="hex colour"):
            validate_color_code("#FFFF", "image_config.color_code")

    def test_rejects_non_hex_digits(self) -> None:
        with pytest.raises(ValueError, match="hex colour"):
            validate_color_code("#GGGGGG", "image_config.color_code")


class TestValidateImage:
    def test_returns_the_tensor(self) -> None:
        tensor = torch.zeros((1, 4, 4, 3))

        assert validate_image(tensor, "image_config.image") is tensor

    def test_rejects_anything_else(self) -> None:
        with pytest.raises(ValueError, match="IMAGE tensor"):
            validate_image("image", "image_config.image")


class TestValidateLatent:
    def test_returns_the_dict(self) -> None:
        latent = {"samples": torch.zeros((1, 4, 8, 8))}

        assert validate_latent(latent, "image_config.latent") is latent

    def test_rejects_a_dict_without_samples(self) -> None:
        with pytest.raises(ValueError, match="holding 'samples'"):
            validate_latent({"other": 1}, "image_config.latent")

    def test_rejects_samples_that_is_not_a_tensor(self) -> None:
        with pytest.raises(ValueError, match="'samples' is a tensor"):
            validate_latent({"samples": "latent"}, "image_config.latent")


class TestValidateLoraList:
    def test_returns_the_list(self) -> None:
        loras = [LoraSpec("detail.safetensors", "", 1.0, 1.0)]

        assert validate_lora_list(loras, "loras") is loras

    def test_accepts_an_empty_list(self) -> None:
        assert validate_lora_list([], "loras") == []

    def test_rejects_a_non_list(self) -> None:
        with pytest.raises(ValueError, match="list of LoraSpec"):
            validate_lora_list("detail.safetensors", "loras")

    def test_rejects_a_list_of_other_objects(self) -> None:
        with pytest.raises(ValueError, match="list of LoraSpec"):
            validate_lora_list(["detail.safetensors"], "loras")