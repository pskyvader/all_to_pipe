import sys
import os
import pytest

# --- 1. Environment Setup ---
COMFY_ROOT = os.path.abspath(r"E:\ComfyUI")
CUSTOM_NODES_DIR = os.path.join(COMFY_ROOT, "custom_nodes")

for path in [COMFY_ROOT, CUSTOM_NODES_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import the correct loader for checkpoints
from nodes import CheckpointLoaderSimple
from all_to_pipe.alltopipe_types.prompts import PromptProcessor

CLIP_L_DIM = 768


# --- 2. Fixture (Loading from Checkpoint) ---
@pytest.fixture(scope="session")
def clip():
    import folder_paths

    # Define the relative path from the models/checkpoints folder
    # Note: ComfyUI expects the path relative to the root checkpoints directory
    checkpoint_name = os.path.join("SD15", "cyberrealistic_v90.safetensors")

    print(f"\n[INIT] Loading CLIP from checkpoint: {checkpoint_name}...")

    try:
        loader = CheckpointLoaderSimple()
        # CheckpointLoaderSimple.load_checkpoint returns (model, clip, vae)
        _, clip_data, _ = loader.load_checkpoint(checkpoint_name)
        return clip_data
    except Exception as e:
        pytest.exit(
            f"Failed to load checkpoint. Ensure the file exists at E:\\ComfyUI\\models\\checkpoints\\{checkpoint_name}\nError: {e}"
        )


# --- 3. The Tests ---
@pytest.mark.parametrize(
    "name, prompt, expected_segments",
    [
        ("Very Short", "a digital painting of a cat", 1),
        ("Boundary Case (75 tokens)", "quality, " * 74 + "quality", 1),
        ("Double Segment", "masterpiece, " * 80, 2),
        ("Monster Prompt", "cinematic, " * 200, 3),
        (
            "Mixed Case & Commas",
            "Masterpiece, best quality, 8k, highly detailed, sharp focus",
            1,
        ),
    ],
)
def test_prompt_processor_various_cases(name, prompt, expected_segments, clip):
    print(f"\n[TEST] Running: {name}")

    # This calls your PromptProcessor logic
    conditioning = PromptProcessor.encode_prompt(prompt, clip)

    # Validate structure
    assert isinstance(conditioning, list)
    cond_tensor = conditioning[0][0]
    pooled_output = conditioning[0][1].get("pooled_output")

    # Validate Tensors
    assert cond_tensor.shape[2] == CLIP_L_DIM

    # Verify greedy packing: check segment alignment
    actual_segments = cond_tensor.shape[1] // 77
    assert actual_segments == expected_segments

    assert pooled_output is not None
    assert pooled_output.shape == (1, CLIP_L_DIM)

    print(f"✓ {name} passed. Shape: {list(cond_tensor.shape)}")
