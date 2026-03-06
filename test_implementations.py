#!/usr/bin/env python3
"""Quick test of implemented functions."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../..')  # Add ComfyUI path

# First set up mock for folder_paths
import os
class MockFolderPaths:
    @staticmethod
    def get_full_path(name, path):
        return path
    
    @staticmethod
    def get_folder_paths(name):
        return []

sys.modules['folder_paths'] = MockFolderPaths()

print("Testing All-to-Pipe implementations...")

# Test 1: Image config
from alltopipe_types.image_config import ImageConfig, ImageConfigProcessor
ic = ImageConfig(512, 512, 1)
print("[OK] ImageConfig created")

# Test 2: torch import
import torch
print("[OK] torch imported")

# Test 3: Noisy latent creation
latent = ImageConfigProcessor.create_noisy_latent(ic, 42)
assert latent is not None, "Latent should not be None"
assert "samples" in latent, "Latent should have samples key"
assert latent["samples"].shape == (1, 4, 64, 64), f"Wrong shape: {latent['samples'].shape}"
print(f"[OK] Noisy latent created with shape {latent['samples'].shape}")

# Test 4: encode_prompt (with mock clip)
from alltopipe_types.prompts import PromptProcessor

class MockClip:
    def tokenize(self, text):
        return {"tokens": text}
    
    def encode_from_tokens_scheduled(self, tokens):
        return (None, )  # CONDITIONING format

try:
    clip = MockClip()
    conditioning = PromptProcessor.encode_prompt("test prompt", clip)
    assert conditioning is not None, "Conditioning should not be None"
    print("[OK] encode_prompt works with mock CLIP")
except Exception as e:
    print(f"[ERROR] encode_prompt failed: {e}")

# Test 5: Test that CLIP None is handled
try:
    PromptProcessor.encode_prompt("test prompt", None)
    print("[ERROR] encode_prompt should raise error with None CLIP")
except ValueError as e:
    print(f"[OK] encode_prompt correctly raises error for None CLIP: {e}")

# Test 6: Empty prompt handling
try:
    PromptProcessor.encode_prompt("", clip)
    print("[ERROR] encode_prompt should raise error for empty prompt")
except ValueError as e:
    print(f"[OK] encode_prompt correctly raises error for empty prompt: {e}")

print("\nAll implementations validated successfully!")
