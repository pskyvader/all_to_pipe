"""
Unit tests for utils module.

Tests utility functions like deep_copy_pipe.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.utils import deep_copy_pipe
from alltopipe_types import Pipe, Model, LoraSpec, Parameters, ImageConfig


class TestUtils(unittest.TestCase):
    """Test suite for utility functions."""

    def test_deep_copy_pipe_empty(self):
        """Test deep copying an empty Pipe."""
        pipe = Pipe()
        copied = deep_copy_pipe(pipe)

        self.assertIsNotNone(copied)
        self.assertIsNot(copied, pipe)
        self.assertIsNone(copied.model)
        self.assertEqual(copied.loras, [])

    def test_deep_copy_pipe_with_model(self):
        """Test deep copying Pipe with model."""
        model = Model(name="test.safetensors", subfolder="checkpoints")
        pipe = Pipe(model=model)

        copied = deep_copy_pipe(pipe)

        self.assertIsNotNone(copied.model)
        self.assertEqual(copied.model.name, "test.safetensors")
        self.assertIsNot(copied.model, pipe.model)

    def test_deep_copy_pipe_with_loras(self):
        """Test deep copying Pipe with LoRAs."""
        lora1 = LoraSpec(
            name="lora1.safetensors",
            subfolder="loras",
            weight=0.8,
            clip_weight=0.5,
        )
        pipe = Pipe(loras=[lora1])

        copied = deep_copy_pipe(pipe)

        self.assertEqual(len(copied.loras), 1)
        self.assertEqual(copied.loras[0].name, "lora1.safetensors")
        self.assertIsNot(copied.loras[0], lora1)

    def test_deep_copy_pipe_with_parameters(self):
        """Test deep copying Pipe with parameters."""
        params = Parameters(
            steps=20,
            cfg=7.5,
            sampler="euler",
            scheduler="karras",
            seed=12345,
        )
        pipe = Pipe(parameters=params)

        copied = deep_copy_pipe(pipe)

        self.assertIsNotNone(copied.parameters)
        self.assertEqual(copied.parameters.steps, 20)
        self.assertIsNot(copied.parameters, pipe.parameters)

    def test_deep_copy_pipe_with_image_config(self):
        """Test deep copying Pipe with image config."""
        config = ImageConfig(width=512, height=768, batch_size=2)
        pipe = Pipe(image_config=config)

        copied = deep_copy_pipe(pipe)

        self.assertIsNotNone(copied.image_config)
        self.assertEqual(copied.image_config.width, 512)
        self.assertIsNot(copied.image_config, pipe.image_config)

    def test_deep_copy_pipe_full(self):
        """Test deep copying completely configured Pipe."""
        model = Model(name="test.safetensors", subfolder="checkpoints")
        lora = LoraSpec(
            name="style.safetensors",
            subfolder="loras",
            weight=0.8,
            clip_weight=0.5,
        )
        params = Parameters(
            steps=25,
            cfg=8.0,
            sampler="dpmpp_2m_karras",
            scheduler="karras",
            seed=42,
        )
        config = ImageConfig(width=768, height=512, batch_size=1)

        pipe = Pipe(
            model=model,
            loras=[lora],
            parameters=params,
            image_config=config,
        )

        copied = deep_copy_pipe(pipe)

        self.assertIsNot(copied, pipe)
        self.assertEqual(copied.model.name, pipe.model.name)
        self.assertEqual(len(copied.loras), len(pipe.loras))
        self.assertEqual(copied.parameters.steps, pipe.parameters.steps)
        self.assertEqual(copied.image_config.width, pipe.image_config.width)


if __name__ == "__main__":
    unittest.main()
