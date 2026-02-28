"""
Unit tests for Pipe type.

Tests the central Pipe container class.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import Model, LoraSpec, Parameters, ImageConfig, Pipe


class TestPipe(unittest.TestCase):
    """Test suite for Pipe type."""

    def test_pipe_creation(self):
        """Test creating a Pipe instance."""
        pipe = Pipe()
        self.assertIsNone(pipe.model)
        self.assertEqual(pipe.loras, [])
        self.assertIsNone(pipe.parameters)
        self.assertIsNone(pipe.image_config)
        self.assertIsNotNone(pipe.positive_prompt)
        self.assertIsNotNone(pipe.negative_prompt)

    def test_pipe_with_model(self):
        """Test Pipe with model."""
        model = Model(name="test.safetensors", subfolder="checkpoints")
        pipe = Pipe(model=model)
        self.assertIsNotNone(pipe.model)
        self.assertEqual(pipe.model.name, "test.safetensors")

    def test_pipe_with_parameters(self):
        """Test Pipe with parameters."""
        params = Parameters(
            steps=20,
            cfg=7.5,
            sampler="euler",
            scheduler="karras",
            seed=12345,
        )
        pipe = Pipe(parameters=params)
        self.assertIsNotNone(pipe.parameters)
        self.assertEqual(pipe.parameters.steps, 20)
        self.assertEqual(pipe.parameters.cfg, 7.5)

    def test_pipe_with_image_config(self):
        """Test Pipe with image config."""
        config = ImageConfig(width=512, height=768, batch_size=2)
        pipe = Pipe(image_config=config)
        self.assertIsNotNone(pipe.image_config)
        self.assertEqual(pipe.image_config.width, 512)

    def test_pipe_with_loras(self):
        """Test Pipe with multiple LoRAs."""
        lora1 = LoraSpec(
            name="lora1.safetensors",
            subfolder="loras",
            weight=0.8,
            clip_weight=0.5,
        )
        lora2 = LoraSpec(
            name="lora2.safetensors",
            subfolder="loras",
            weight=0.6,
            clip_weight=0.4,
        )
        pipe = Pipe(loras=[lora1, lora2])
        self.assertEqual(len(pipe.loras), 2)
        self.assertEqual(pipe.loras[0].name, "lora1.safetensors")
        self.assertEqual(pipe.loras[1].name, "lora2.safetensors")

    def test_pipe_companion_data(self):
        """Test Pipe companion file data attributes."""
        pipe = Pipe()

        # Test setting companion model data
        pipe.companion_model_data = {"sampler": "euler", "steps": [20, 30]}
        self.assertIsNotNone(pipe.companion_model_data)
        self.assertEqual(pipe.companion_model_data["sampler"], "euler")

        # Test setting companion lora data
        pipe.companion_lora_data = [
            {"name": "lora1", "weight": 0.8},
            {"name": "lora2", "weight": 0.6},
        ]
        self.assertIsNotNone(pipe.companion_lora_data)
        self.assertEqual(len(pipe.companion_lora_data), 2)

    def test_pipe_full_configuration(self):
        """Test Pipe with complete configuration."""
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

        self.assertIsNotNone(pipe.model)
        self.assertEqual(len(pipe.loras), 1)
        self.assertIsNotNone(pipe.parameters)
        self.assertIsNotNone(pipe.image_config)
        self.assertEqual(pipe.image_config.width, 768)


if __name__ == "__main__":
    unittest.main()
