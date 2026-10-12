"""
Unit tests for the element registry.

Covers the three things the single-element nodes depend on: the key list
matches the shape of a Pipe, reading reports an unset path instead of
returning nothing, and writing leaves the incoming Pipe untouched while
repairing whatever the change invalidated.
"""

import sys
from pathlib import Path

import pytest
import torch

# The package lives under ComfyUI, so the repo root is what has to be on the
# path for `custom_nodes.all_to_pipe` to resolve.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from custom_nodes.all_to_pipe.alltopipe_types import (
    ImageConfig,
    LoraSpec,
    Model,
    Parameters,
    Pipe,
    PositivePrompt,
    Template,
)
from custom_nodes.all_to_pipe.common.elements import ElementRegistry

VAE: object = object()


def make_pipe() -> Pipe:
    """A Pipe with every container set, so reads have something to resolve."""
    pipe = Pipe(
        model=Model(name="sd15.safetensors", subfolder="", clip_skip=-1),
        loras=[LoraSpec("detail.safetensors", "", 0.8, 0.5)],
        parameters=Parameters(20, 7.0, "euler", "normal", 42, 1.0),
        image_config=ImageConfig(512, 768, 1, 1.0, "#FF0000"),
        positive_template=Template(
            "positive", ["age"], "a <age> cat", True, 0.0, 0.5
        ),
        negative_template=Template("negative", [], "blurry", False, 0.1, 0.4),
        positive_prompt=PositivePrompt(),
    )
    pipe.positive_prompt.age = "young"
    return pipe


class TestElementList:
    def test_covers_every_container(self) -> None:
        keys = ElementRegistry.export_keys()

        for container in (
            "model",
            "loras",
            "parameters",
            "image_config",
            "positive_template",
            "negative_template",
            "positive_prompt",
            "negative_prompt",
        ):
            # `loras` itself is the list, everything else is a field on it.
            assert any(
                key == container or key.startswith(f"{container}.") for key in keys
            ), container

    def test_lists_every_allowed_prompt_feature(self) -> None:
        keys = ElementRegistry.export_keys()

        for feature in PositivePrompt.ALLOWED_FEATURES:
            assert f"positive_prompt.{feature}" in keys

    def test_derived_fields_are_read_only(self) -> None:
        assert "positive_template.parsed_template" in ElementRegistry.export_keys()
        assert "positive_template.parsed_template" not in ElementRegistry.update_keys()
        assert "negative_template.placeholders" not in ElementRegistry.update_keys()

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an element"):
            ElementRegistry.get("parameters.nonexistent")


class TestRead:
    def test_reads_a_scalar_field(self) -> None:
        assert ElementRegistry.read(make_pipe(), "parameters.steps") == 20

    def test_reads_through_a_container(self) -> None:
        assert ElementRegistry.read(make_pipe(), "positive_prompt.age") == "young"

    def test_reads_a_list_entry(self) -> None:
        assert ElementRegistry.read(make_pipe(), "loras[0].weight") == 0.8

    def test_unset_field_is_reported(self) -> None:
        pipe = make_pipe()

        with pytest.raises(ValueError, match="latent"):
            ElementRegistry.read(pipe, "image_config.latent")

    def test_missing_container_is_reported(self) -> None:
        pipe = Pipe()

        with pytest.raises(ValueError, match="model"):
            ElementRegistry.read(pipe, "model.name")

    def test_empty_list_is_reported(self) -> None:
        pipe = make_pipe()
        pipe.loras = []

        with pytest.raises(ValueError, match="entry 0 is missing"):
            ElementRegistry.read(pipe, "loras[0].weight")


class TestWrite:
    def test_sets_a_scalar_field(self) -> None:
        pipe = ElementRegistry.write(make_pipe(), "parameters.steps", 35)

        assert pipe.parameters is not None
        assert pipe.parameters.steps == 35

    def test_leaves_the_incoming_pipe_alone(self) -> None:
        original = make_pipe()
        original.parameters.seed = 7

        updated = ElementRegistry.write(original, "parameters.seed", 99)

        assert original.parameters is not None
        assert original.parameters.seed == 7
        assert updated.parameters is not None
        assert updated.parameters.seed == 99
        assert updated.parameters is not original.parameters

    def test_sets_a_list_entry_without_touching_the_original(self) -> None:
        original = make_pipe()
        original.loras[0].cached_lora = {"weight": 1}

        updated = ElementRegistry.write(original, "loras[0].weight", 1.5)

        assert original.loras[0].weight == 0.8
        assert updated.loras[0].weight == 1.5

    def test_creates_a_missing_container(self) -> None:
        pipe = ElementRegistry.write(Pipe(), "parameters.cfg", 4.5)

        assert pipe.parameters is not None
        assert pipe.parameters.cfg == 4.5
        # The fields that were not written keep the defaults of the container.
        assert pipe.parameters.steps == 20

    def test_creates_a_missing_prompt_field(self) -> None:
        pipe = Pipe()
        pipe.positive_prompt = None

        updated = ElementRegistry.write(pipe, "positive_prompt.hair", "red")

        assert updated.positive_prompt is not None
        assert updated.positive_prompt.hair == "red"

    def test_creates_a_template_when_writing_its_text(self) -> None:
        updated = ElementRegistry.write(
            Pipe(), "positive_template.text", "a <age> <hair> cat"
        )

        assert updated.positive_template is not None
        assert updated.positive_template.text == "a <age> <hair> cat"
        assert updated.positive_template.type == "positive"
        assert updated.positive_template.placeholders == ["age", "hair"]
        assert updated.positive_template.parsed_template is None

    def test_refuses_other_template_fields_without_a_template(self) -> None:
        with pytest.raises(ValueError, match="no positive_template"):
            ElementRegistry.write(Pipe(), "positive_template.decay", 0.5)

    def test_refuses_a_container_it_cannot_build(self) -> None:
        with pytest.raises(ValueError, match="no model"):
            ElementRegistry.write(Pipe(), "model.clip_skip", -2)

    def test_refuses_a_read_only_field(self) -> None:
        with pytest.raises(ValueError, match="derived"):
            ElementRegistry.write(make_pipe(), "positive_template.parsed_template", "x")

    def test_empty_list_is_reported(self) -> None:
        pipe = make_pipe()
        pipe.loras = []

        with pytest.raises(ValueError, match="no loras"):
            ElementRegistry.write(pipe, "loras[0].weight", 1.0)


class TestWriteRepairs:
    def test_prompt_change_invalidates_the_parsed_template(self) -> None:
        pipe = make_pipe()
        pipe.positive_template.parsed_template = "a young cat"

        updated = ElementRegistry.write(pipe, "positive_prompt.age", "old")

        assert updated.positive_template.parsed_template is None

    def test_template_text_rebuilds_placeholders(self) -> None:
        pipe = make_pipe()
        pipe.positive_template.parsed_template = "a young cat"

        updated = ElementRegistry.write(pipe, "positive_template.text", "a <age> <hair> cat")

        assert updated.positive_template.placeholders == ["age", "hair"]
        assert updated.positive_template.parsed_template is None

    def test_resize_drops_the_stale_image_and_latent(self) -> None:
        pipe = make_pipe()
        pipe.image_config.image = "image"
        pipe.image_config.latent = {"samples": "latent"}

        updated = ElementRegistry.write(pipe, "image_config.width", 1024)

        assert updated.image_config.width == 1024
        assert updated.image_config.image is None
        assert updated.image_config.latent is None

    def test_new_image_without_a_vae_leaves_the_latent_unset(self) -> None:
        pipe = make_pipe()
        image = torch.zeros((1, 4, 4, 3))

        updated = ElementRegistry.write(pipe, "image_config.image", image)

        assert updated.image_config.image is image
        assert updated.image_config.latent is None

    def test_new_model_drops_the_cached_checkpoint(self) -> None:
        pipe = make_pipe()
        pipe.model.cached_model = ("model", "clip", VAE)

        updated = ElementRegistry.write(pipe, "model.name", "sdxl.safetensors")

        assert updated.model.name == "sdxl.safetensors"
        assert updated.model.cached_model is None

    def test_new_lora_weight_drops_the_cached_tensors(self) -> None:
        pipe = make_pipe()
        pipe.loras[0].cached_lora = {"weight": 1}
        pipe.loras[0].cached_metadata = {"ss_output_name": "detail"}

        updated = ElementRegistry.write(pipe, "loras[0].weight", 1.5)

        assert updated.loras[0].cached_lora is None
        assert updated.loras[0].cached_metadata is None
        assert pipe.loras[0].cached_lora == {"weight": 1}


class TestWriteValidates:
    """Every writable field is checked, and a rejected value changes nothing."""

    def test_out_of_range_number_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            ElementRegistry.write(make_pipe(), "parameters.steps", 500)

    def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be an INT"):
            ElementRegistry.write(make_pipe(), "parameters.steps", "many")

    def test_sampler_outside_the_choices_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            ElementRegistry.write(make_pipe(), "parameters.sampler", "nope")

    def test_clip_skip_must_stay_negative(self) -> None:
        with pytest.raises(ValueError, match="between -24 and -1"):
            ElementRegistry.write(make_pipe(), "model.clip_skip", 0)

    def test_lora_weight_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="between -5.0 and 5.0"):
            ElementRegistry.write(make_pipe(), "loras[0].weight", 9.0)

    def test_image_must_be_a_tensor(self) -> None:
        with pytest.raises(ValueError, match="IMAGE tensor"):
            ElementRegistry.write(make_pipe(), "image_config.image", "image")

    def test_latent_must_hold_samples(self) -> None:
        with pytest.raises(ValueError, match="holding 'samples'"):
            ElementRegistry.write(make_pipe(), "image_config.latent", {"other": 1})

    def test_template_text_cannot_be_blank(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ElementRegistry.write(make_pipe(), "positive_template.text", "   ")

    def test_prompt_switch_must_be_a_boolean(self) -> None:
        with pytest.raises(ValueError, match="must be a BOOLEAN"):
            ElementRegistry.write(make_pipe(), "positive_prompt.allow_missing", 1)

    def test_bad_colour_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hex colour"):
            ElementRegistry.write(make_pipe(), "image_config.color_code", "#FFFF")

    def test_lora_list_must_hold_specs(self) -> None:
        with pytest.raises(ValueError, match="list of LoraSpec"):
            ElementRegistry.write(make_pipe(), "loras", ["detail.safetensors"])

    def test_rejected_value_leaves_the_pipe_alone(self) -> None:
        original = make_pipe()

        with pytest.raises(ValueError):
            ElementRegistry.write(original, "parameters.steps", 500)

        assert original.parameters is not None
        assert original.parameters.steps == 20

    def test_an_int_is_widened_for_a_float_field(self) -> None:
        updated = ElementRegistry.write(make_pipe(), "parameters.cfg", 4)

        assert updated.parameters is not None
        assert isinstance(updated.parameters.cfg, float)
        assert updated.parameters.cfg == 4.0

    def test_colour_is_normalised(self) -> None:
        updated = ElementRegistry.write(make_pipe(), "image_config.color_code", "f00")

        assert updated.image_config.color_code == "#FF0000"

    def test_blank_colour_becomes_a_random_one(self) -> None:
        updated = ElementRegistry.write(make_pipe(), "image_config.color_code", "")

        assert updated.image_config.color_code is None