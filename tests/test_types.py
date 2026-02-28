"""
Unit tests for All-to-Pipe types.

Tests the core type classes:
- Model
- LoraSpec
- Parameters
- ImageConfig
- PositivePrompt
- NegativePrompt
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alltopipe_types import (
    Model,
    LoraSpec,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
    Pipe,
)


class TestModel(unittest.TestCase):
    """Test suite for Model type."""

    def test_model_creation(self):
        """Test creating a Model instance."""
        model = Model(name="model.safetensors", subfolder="checkpoints")
        self.assertEqual(model.name, "model.safetensors")
        self.assertEqual(model.subfolder, "checkpoints")

    def test_model_attributes(self):
        """Test that Model has expected attributes."""
        model = Model(name="test_model", subfolder="test_folder")
        self.assertTrue(hasattr(model, "name"))
        self.assertTrue(hasattr(model, "subfolder"))


class TestLoraSpec(unittest.TestCase):
    """Test suite for LoraSpec type."""

    def test_lora_creation(self):
        """Test creating a LoraSpec instance."""
        lora = LoraSpec(
            name="lora.safetensors",
            subfolder="loras",
            weight=0.8,
            clip_weight=0.5,
        )
        self.assertEqual(lora.name, "lora.safetensors")
        self.assertEqual(lora.subfolder, "loras")
        self.assertEqual(lora.weight, 0.8)
        self.assertEqual(lora.clip_weight, 0.5)

    def test_lora_weight_range(self):
        """Test that LoRA weights are properly stored."""
        lora = LoraSpec(
            name="test.safetensors",
            subfolder="loras",
            weight=1.5,  # Can be > 1.0
            clip_weight=-0.5,  # Can be negative
        )
        self.assertEqual(lora.weight, 1.5)
        self.assertEqual(lora.clip_weight, -0.5)


class TestParameters(unittest.TestCase):
    """Test suite for Parameters type."""

    def test_parameters_creation(self):
        """Test creating a Parameters instance."""
        params = Parameters(
            steps=20,
            cfg=7.5,
            sampler="euler",
            scheduler="karras",
            seed=12345,
        )
        self.assertEqual(params.steps, 20)
        self.assertEqual(params.cfg, 7.5)
        self.assertEqual(params.sampler, "euler")
        self.assertEqual(params.scheduler, "karras")
        self.assertEqual(params.seed, 12345)

    def test_parameters_values(self):
        """Test various parameter values."""
        params = Parameters(
            steps=100,
            cfg=15.0,
            sampler="dpm++_2m_karras",
            scheduler="exponential",
            seed=0,
        )
        self.assertEqual(params.steps, 100)
        self.assertEqual(params.cfg, 15.0)
        self.assertIn("dpm", params.sampler.lower())


class TestImageConfig(unittest.TestCase):
    """Test suite for ImageConfig type."""

    def test_image_config_creation(self):
        """Test creating an ImageConfig instance."""
        config = ImageConfig(
            width=512,
            height=768,
            batch_size=2,
            noise=0.8,
            color_code="#FF0000",
        )
        self.assertEqual(config.width, 512)
        self.assertEqual(config.height, 768)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.noise, 0.8)
        self.assertEqual(config.color_code, "#FF0000")

    def test_image_config_noise_clamping(self):
        """Test that noise is clamped to 0-1 range."""
        # Test noise > 1.0 is clamped
        config = ImageConfig(width=512, height=512, batch_size=1, noise=2.0)
        self.assertEqual(config.noise, 1.0)

        # Test noise < 0 is clamped
        config = ImageConfig(width=512, height=512, batch_size=1, noise=-0.5)
        self.assertEqual(config.noise, 0.0)

        # Test valid noise
        config = ImageConfig(width=512, height=512, batch_size=1, noise=0.5)
        self.assertEqual(config.noise, 0.5)

    def test_image_config_optional_color(self):
        """Test ImageConfig with optional color code."""
        config = ImageConfig(width=512, height=512, batch_size=1, color_code=None)
        self.assertIsNone(config.color_code)


class TestPrompts(unittest.TestCase):
    """Test suite for PositivePrompt and NegativePrompt types."""

    def test_positive_prompt_creation(self):
        """Test creating a PositivePrompt instance."""
        prompt = PositivePrompt()
        self.assertIsNone(prompt.age)
        self.assertIsNone(prompt.body)
        self.assertIsNone(prompt.clothes)

    def test_positive_prompt_attributes(self):
        """Test PositivePrompt allowed attributes."""
        prompt = PositivePrompt()
        prompt.age = "young"
        prompt.body = "slim"
        prompt.clothes = "casual"
        prompt.background = "outdoor"
        prompt.face = "smiling"
        prompt.lora = "lora_style"
        prompt.model = "model_style"

        self.assertEqual(prompt.age, "young")
        self.assertEqual(prompt.body, "slim")
        self.assertEqual(prompt.lora, "lora_style")

    def test_negative_prompt_creation(self):
        """Test creating a NegativePrompt instance."""
        prompt = NegativePrompt()
        self.assertIsNone(prompt.static)
        self.assertIsNone(prompt.lora)
        self.assertIsNone(prompt.model)

    def test_negative_prompt_attributes(self):
        """Test NegativePrompt allowed attributes."""
        prompt = NegativePrompt()
        prompt.static = "blurry"
        prompt.lora = "bad_lora"
        prompt.model = "bad_model"

        self.assertEqual(prompt.static, "blurry")
        self.assertEqual(prompt.lora, "bad_lora")


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


if __name__ == "__main__":
    unittest.main()
