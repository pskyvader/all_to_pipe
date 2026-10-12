"""
Unit tests for the export single node.

Covers the key list the node offers and what it hands back for a key.
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
from custom_nodes.all_to_pipe.nodes.export_single_node import ExportSingleNode


def make_pipe() -> Pipe:
    """A Pipe with every container and every prompt field set."""
    pipe = Pipe(
        model=Model("sd15.safetensors", "", -1),
        loras=[LoraSpec("detail.safetensors", "", 0.8, 0.5)],
        parameters=Parameters(20, 7.0, "euler", "normal", 42, 1.0),
        image_config=ImageConfig(512, 512, 1, 1.0, "#FF0000"),
        positive_template=Template("positive", ["age"], "a <age> cat", True, 0.0, 0.5),
        negative_template=Template("negative", [], "blurry", False, 0.0, 0.5),
    )
    pipe.image_config.image = torch.zeros((1, 4, 4, 3))
    pipe.image_config.latent = {"samples": torch.zeros((1, 4, 8, 8))}
    pipe.positive_template.parsed_template = "a age cat"
    pipe.negative_template.parsed_template = "blurry"

    for prompt, features in (
        (pipe.positive_prompt, PositivePrompt.ALLOWED_FEATURES),
        (pipe.negative_prompt, NegativePrompt.ALLOWED_FEATURES),
    ):
        for feature in features:
            setattr(prompt, feature, f"a {feature}")
        prompt.model = "companion text"
        prompt.lora = "companion text"

    return pipe


class TestInputTypes:
    def test_offers_every_element(self) -> None:
        key_input = ExportSingleNode.INPUT_TYPES()["required"]["key"]

        assert key_input[0] == ElementRegistry.export_keys()
        assert "parameters.steps" in key_input[0]
        assert "image_config.width" in key_input[0]

    def test_takes_a_pipe(self) -> None:
        assert ExportSingleNode.INPUT_TYPES()["required"]["pipe"] == ("PIPE",)


class TestExecute:
    def test_exports_the_selected_field(self) -> None:
        assert ExportSingleNode.execute(make_pipe(), "parameters.steps") == (20,)

    def test_exports_a_string_field(self) -> None:
        assert ExportSingleNode.execute(make_pipe(), "image_config.color_code") == (
            "#FF0000",
        )

    def test_exports_a_list_entry(self) -> None:
        assert ExportSingleNode.execute(make_pipe(), "loras[0].weight") == (0.8,)

    def test_every_offered_key_resolves(self) -> None:
        pipe = make_pipe()

        for key in ElementRegistry.export_keys():
            value, = ExportSingleNode.execute(pipe, key)
            assert value is not None, key

    def test_unset_field_raises(self) -> None:
        pipe = make_pipe()
        pipe.image_config.latent = None

        with pytest.raises(ValueError, match="latent"):
            ExportSingleNode.execute(pipe, "image_config.latent")

    def test_missing_container_raises(self) -> None:
        pipe = make_pipe()
        pipe.model = None

        with pytest.raises(ValueError, match="model"):
            ExportSingleNode.execute(pipe, "model.name")

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not an element"):
            ExportSingleNode.execute(make_pipe(), "parameters.missing")