"""
Unit tests for the update pipe node.

Covers the key list the node offers and the guarantee that an update lands on
the returned Pipe only.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from custom_nodes.all_to_pipe.alltopipe_types import (
    ImageConfig,
    LoraSpec,
    Model,
    NegativePrompt,
    Parameters,
    Pipe,
    PositivePrompt,
    Template,
)
from custom_nodes.all_to_pipe.common.elements import ElementRegistry
from custom_nodes.all_to_pipe.nodes.update_pipe_node import UpdatePipeNode

# Keys whose field only accepts one of a fixed set of values, so the round trip
# below cannot simply nudge the value the Pipe already holds.
REPLACEMENTS: dict[str, object] = {
    "parameters.sampler": "ddim",
    "parameters.scheduler": "karras",
    "image_config.color_code": "#0000FF",
}


def make_pipe() -> Pipe:
    """A Pipe with every container and every prompt field set."""
    pipe = Pipe(
        model=Model("sd15.safetensors", "", -2),
        loras=[LoraSpec("detail.safetensors", "", 0.8, 0.5)],
        parameters=Parameters(20, 7.0, "euler", "normal", 42, 0.5),
        image_config=ImageConfig(512, 512, 1, 0.5, "#FF0000"),
        positive_template=Template("positive", ["age"], "a <age> cat", True, 0.0, 0.5),
        negative_template=Template("negative", [], "blurry", False, 0.0, 0.5),
    )
    pipe.image_config.image = torch.zeros((1, 4, 4, 3))
    pipe.image_config.latent = {"samples": torch.zeros((1, 4, 8, 8))}

    for prompt, features in (
        (pipe.positive_prompt, PositivePrompt.ALLOWED_FEATURES),
        (pipe.negative_prompt, NegativePrompt.ALLOWED_FEATURES),
    ):
        for feature in features:
            setattr(prompt, feature, f"a {feature}")
        prompt.model = "companion text"
        prompt.lora = "companion text"

    return pipe


def changed(value: object) -> object:
    """A value of the same shape as the one already held, so a round trip works."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return f"{value} updated"
    return value


class TestInputTypes:
    def test_offers_every_writable_element(self) -> None:
        key_input = UpdatePipeNode.INPUT_TYPES()["required"]["key"]

        assert key_input[0] == ElementRegistry.update_keys()
        assert "parameters.cfg" in key_input[0]
        assert "positive_prompt.style" in key_input[0]

    def test_keeps_the_pipe_optional(self) -> None:
        inputs = UpdatePipeNode.INPUT_TYPES()

        assert inputs["optional"]["pipe"] == ("PIPE",)
        assert inputs["required"]["value"] == "*"


class TestExecute:
    def test_sets_the_selected_field(self) -> None:
        (pipe,) = UpdatePipeNode.execute("parameters.cfg", 4.5, make_pipe())

        assert pipe.parameters is not None
        assert pipe.parameters.cfg == 4.5

    def test_builds_a_pipe_when_none_is_connected(self) -> None:
        (pipe,) = UpdatePipeNode.execute("image_config.width", 1024)

        assert isinstance(pipe, Pipe)
        assert pipe.image_config is not None
        assert pipe.image_config.width == 1024

    def test_does_not_touch_the_incoming_pipe(self) -> None:
        original = make_pipe()

        (updated,) = UpdatePipeNode.execute("parameters.seed", 99, original)

        assert original.parameters is not None
        assert original.parameters.seed == 42
        assert updated.parameters is not None
        assert updated.parameters.seed == 99

    def test_every_offered_key_writes_and_reads_back(self) -> None:
        pipe = make_pipe()

        for key in ElementRegistry.update_keys():
            current = ElementRegistry.read(pipe, key)
            replacement = REPLACEMENTS.get(key, changed(current))

            (updated,) = UpdatePipeNode.execute(key, replacement, pipe)

            written = ElementRegistry.read(updated, key)
            if isinstance(replacement, torch.Tensor):
                assert torch.equal(written, replacement), key
            else:
                assert written == replacement, key

    def test_a_rejected_value_is_not_written(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            UpdatePipeNode.execute("parameters.steps", 500, make_pipe())

    def test_read_only_key_raises(self) -> None:
        with pytest.raises(ValueError, match="derived"):
            UpdatePipeNode.execute("positive_template.parsed_template", "x", make_pipe())

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not an element"):
            UpdatePipeNode.execute("parameters.missing", 1, make_pipe())