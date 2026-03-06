"""
ImageConfig type and processor for All-to-Pipe.

Handles image dimensions, batch size, and latent generation.
"""

from typing import Optional, Any, Tuple
import random
import torch


class ImageConfig:
    """
    Image configuration for generation.
    Specifies dimensions, batch size, and noise parameters.
    """

    def __init__(
        self,
        width: int,
        height: int,
        batch_size: int,
        noise: float = 1.0,
        color_code: Optional[str] = None,
    ) -> None:
        """
        Initialize ImageConfig.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            batch_size: Number of images in batch
            noise: Noise level (0.0-1.0 percentage)
            color_code: Hex color code (e.g., "#FF0000") or None for random
        """
        self.width: int = width
        self.height: int = height
        self.batch_size: int = batch_size
        self.noise: float = max(0.0, min(1.0, noise))  # Clamp 0-1
        self.color_code: Optional[str] = color_code


class ImageConfigProcessor:
    """
    Processor for ImageConfig operations.
    Handles creation of initial noisy latent images.
    """

    @staticmethod
    def create_noisy_latent(
        image_config: ImageConfig,
        seed: int,
    ) -> Optional[Any]:
        """
        Create a batch of noisy latent images based on config.

        Args:
            image_config: ImageConfig instance with dimensions and noise settings
            seed: Random seed for reproducibility

        Returns:
            Latent tensor ready for KSampler denoise
            or None if creation fails

        Raises:
            ValueError: If image_config is invalid
        """
        if image_config is None:
            raise ValueError("ImageConfig cannot be None")

        if image_config.width <= 0 or image_config.height <= 0:
            raise ValueError("Width and height must be positive")

        if image_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)

        # Create noisy latent tensor
        # Latent space is 1/8 the size of the image (height//8, width//8)
        latent_height = image_config.height // 8
        latent_width = image_config.width // 8

        # Create noisy latent with specified noise level
        # noise=1.0 means full noise, noise=0.0 means no noise
        noisy_latent = torch.randn(
            (image_config.batch_size, 4, latent_height, latent_width)
        ) * image_config.noise

        # Return in the ComfyUI latent format
        return {
            "samples": noisy_latent,
            "downscale_ratio_spacial": 8
        }

    @staticmethod
    def get_color_from_code(color_code: Optional[str]) -> Tuple[int, int, int]:
        """
        Parse hex color code to RGB tuple.

        Args:
            color_code: Hex color code like "#FF0000" or None for random

        Returns:
            Tuple of (R, G, B) values (0-255)

        Raises:
            ValueError: If color code format is invalid
        """
        if color_code is None:
            # Random color
            return (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        if not isinstance(color_code, str):
            raise ValueError("Color code must be a string or None")

        color_code = color_code.strip()
        if not color_code.startswith("#"):
            raise ValueError("Color code must start with #")

        color_code = color_code[1:]
        if len(color_code) != 6:
            raise ValueError("Color code must be 6 hex characters")

        try:
            r: int = int(color_code[0:2], 16)
            g: int = int(color_code[2:4], 16)
            b: int = int(color_code[4:6], 16)
            return (r, g, b)
        except ValueError:
            raise ValueError(f"Invalid hex color code: {color_code}")
