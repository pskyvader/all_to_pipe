import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import comfy.lora
import comfy.lora_convert
import comfy.model_patcher
import comfy.sd
import comfy.utils
import comfy.weight_adapter
import folder_paths
import torch

logger: logging.Logger = logging.getLogger("AllToPipe")


@dataclass(frozen=True)
class _CompatibilityProfile:
    family: str | None
    architecture: str | None


class LoraSpec:
    def __init__(
        self,
        name: str,
        subfolder: str,
        weight: float,
        clip_weight: float,
        cached_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name: str = name
        self.subfolder: str = subfolder
        self.weight: float = weight
        self.clip_weight: float = clip_weight
        self.cached_lora: Any | None = None
        self.cached_metadata: dict[str, Any] | None = cached_metadata


class LoraProcessor:
    _VARIANT_FAMILIES: set[str] = {"pony", "illustrious"}
    _FAMILY_TO_ARCHITECTURE: dict[str, str] = {
        "sd15": "sd15",
        "sd20": "sd2x",
        "sd21": "sd2x",
        "sdxl": "sdxl",
        "pony": "sdxl",
        "illustrious": "sdxl",
        "sd3": "sd3",
        "flux": "flux",
    }
    _FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("pony", ("ponyxl", "pony")),
        ("illustrious", ("illustrious", "ilxl", "illxl")),
        ("sd21", ("stablediffusionv21", "sd21")),
        ("sd20", ("stablediffusionv20", "sd20")),
        ("sd15", ("stablediffusionv15", "sd15")),
        ("sdxl", ("stablediffusionxl", "sdxl")),
        ("sd3", ("stablediffusion3", "sd3")),
        ("flux", ("flux",)),
    )

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    @classmethod
    def _family_from_texts(cls, *texts: str | None) -> str | None:
        normalized_texts = [cls._normalize_text(text) for text in texts if text]

        for family, patterns in cls._FAMILY_PATTERNS:
            for text in normalized_texts:
                if any(pattern in text for pattern in patterns):
                    return family
        return None

    @staticmethod
    def _metadata_text_sources(metadata: dict[str, Any] | None) -> list[str]:
        if not metadata:
            return []

        preferred_keys: tuple[str, ...] = (
            "ss_base_model_version",
            "ss_sd_model_name",
            "ss_output_name",
            "modelspec.title",
            "modelspec.architecture",
            "modelspec.implementation",
            "modelspec.predict_key",
            "architecture",
            "base_model_version",
        )

        sources: list[str] = []
        for key in preferred_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value:
                sources.append(value)

        return sources

    @classmethod
    def _family_to_architecture(cls, family: str | None) -> str | None:
        if family is None:
            return None
        return cls._FAMILY_TO_ARCHITECTURE.get(family, family)

    @staticmethod
    def _family_from_model_config(
        model: comfy.model_patcher.ModelPatcher | None,
    ) -> str | None:
        if model is None:
            return None

        model_config = getattr(getattr(model, "model", None), "model_config", None)
        if model_config is None:
            return None

        config_name = type(model_config).__name__.lower()
        if "sd15" in config_name:
            return "sd15"
        if "sd21" in config_name:
            return "sd21"
        if "sd20" in config_name:
            return "sd20"
        if "sdxl" in config_name or config_name in {
            "ssd1b",
            "segmind_vega",
            "koala_700m",
            "koala_1b",
        }:
            return "sdxl"
        if "sd3" in config_name:
            return "sd3"
        if "flux" in config_name:
            return "flux"

        unet_config = getattr(model_config, "unet_config", {}) or {}
        context_dim = unet_config.get("context_dim")
        model_channels = unet_config.get("model_channels")
        adm_in_channels = unet_config.get("adm_in_channels")

        if context_dim == 768 and model_channels == 320:
            return "sd15"
        if context_dim == 1024 and model_channels == 320:
            return "sd20"
        if context_dim == 2048 and model_channels == 320:
            return "sdxl"
        if context_dim == 1280 and model_channels == 384:
            return "sdxl"
        if adm_in_channels in {1536, 2048}:
            return "sd20"
        return None

    @classmethod
    def _family_from_model_spec(cls, model_spec: Any | None) -> str | None:
        if model_spec is None:
            return None
        return cls._family_from_texts(
            getattr(model_spec, "subfolder", None),
            getattr(model_spec, "name", None),
        )

    @classmethod
    def _family_from_lora_spec(cls, lora: LoraSpec) -> str | None:
        return cls._family_from_texts(
            getattr(lora, "subfolder", None),
            getattr(lora, "name", None),
            *cls._metadata_text_sources(getattr(lora, "cached_metadata", None)),
        )

    @staticmethod
    def _architecture_from_model(
        model: comfy.model_patcher.ModelPatcher | None,
    ) -> str | None:
        if model is None:
            return None

        model_config = getattr(getattr(model, "model", None), "model_config", None)
        if model_config is None:
            return None

        config_name = type(model_config).__name__.lower()
        if "sd15" in config_name:
            return "sd15"
        if "sd21" in config_name or "sd20" in config_name:
            return "sd2x"
        if "sdxl" in config_name or config_name in {
            "ssd1b",
            "segmind_vega",
            "koala_700m",
            "koala_1b",
        }:
            return "sdxl"
        if "sd3" in config_name:
            return "sd3"
        if "flux" in config_name:
            return "flux"

        unet_config = getattr(model_config, "unet_config", {}) or {}
        context_dim = unet_config.get("context_dim")
        model_channels = unet_config.get("model_channels")
        adm_in_channels = unet_config.get("adm_in_channels")

        if context_dim == 768 and model_channels == 320:
            return "sd15"
        if context_dim == 1024 and model_channels == 320:
            return "sd2x"
        if context_dim == 2048 and model_channels == 320:
            return "sdxl"
        if context_dim == 1280 and model_channels == 384:
            return "sdxl"
        if adm_in_channels in {1536, 2048}:
            return "sd2x"
        return None

    @classmethod
    def _architecture_from_lora(
        cls, lora_weights: dict[str, Any], lora: LoraSpec
    ) -> str | None:
        family = cls._family_from_lora_spec(lora)
        if family is not None:
            return cls._family_to_architecture(family)

        lora_keys = list(lora_weights.keys())
        if any(
            key.startswith(("lora_te1_", "lora_te2_", "lora_prior_te_"))
            for key in lora_keys
        ):
            return "sdxl"
        return None

    @classmethod
    def _profile_for_model(
        cls,
        model_or_keys: comfy.model_patcher.ModelPatcher | set[str] | None,
        model_spec: Any | None,
    ) -> _CompatibilityProfile:
        family = cls._family_from_model_spec(model_spec)

        model = model_or_keys if isinstance(model_or_keys, comfy.model_patcher.ModelPatcher) else None
        model_keys = model_or_keys if isinstance(model_or_keys, set) else None

        if family is None and model is not None:
            family = cls._family_from_model_config(model)

        architecture = cls._family_to_architecture(family)
        if architecture is None and model is not None:
            architecture = cls._architecture_from_model(model)
        if architecture is None and model_keys is not None:
            architecture = cls._infer_model_architecture(model_keys, model_spec)
        return _CompatibilityProfile(family=family, architecture=architecture)

    @classmethod
    def _profile_for_lora(cls, lora_weights: dict[str, Any], lora: LoraSpec) -> _CompatibilityProfile:
        family = cls._family_from_lora_spec(lora)
        architecture = cls._family_to_architecture(family)
        if architecture is None:
            architecture = cls._architecture_from_lora(lora_weights, lora)
        return _CompatibilityProfile(family=family, architecture=architecture)

    @classmethod
    def _profiles_are_compatible(
        cls, model_profile: _CompatibilityProfile, lora_profile: _CompatibilityProfile
    ) -> bool:
        if model_profile.family and lora_profile.family:
            if model_profile.family != lora_profile.family:
                return False
        elif (
            model_profile.family in cls._VARIANT_FAMILIES
            or lora_profile.family in cls._VARIANT_FAMILIES
        ):
            return False

        if model_profile.architecture and lora_profile.architecture:
            if model_profile.architecture != lora_profile.architecture:
                return False

        return True

    @staticmethod
    def _has_lora_structure(lora_weights: dict[str, Any]) -> bool:
        if not lora_weights:
            return False

        prefixes: tuple[str, ...] = (
            "lora_te_",
            "lora_te1_",
            "lora_te2_",
            "lora_unet_",
            "diffusion_model.",
            "transformer.",
            "base_model.model.",
            "unet.",
            "lycoris_",
        )
        return any(key.startswith(prefixes) for key in lora_weights)

    @staticmethod
    def _build_model_key_map(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP | None = None,
    ) -> dict[str, str]:
        key_map: dict[str, str] = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

        if clip is not None and getattr(clip, "cond_stage_model", None) is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        return key_map

    @staticmethod
    def _infer_model_architecture(
        model_or_keys: comfy.model_patcher.ModelPatcher | set[str] | None,
        model_spec: Any | None = None,
    ) -> str | None:
        family = LoraProcessor._family_from_model_spec(model_spec)
        if family is not None:
            return LoraProcessor._family_to_architecture(family)

        if isinstance(model_or_keys, comfy.model_patcher.ModelPatcher):
            return LoraProcessor._architecture_from_model(model_or_keys)

        model_keys = model_or_keys if isinstance(model_or_keys, set) else set()
        if not model_keys:
            return None

        key_count = len(model_keys)
        key_list = list(model_keys)[:100]
        has_input_blocks = any("input_blocks" in key for key in key_list)
        has_diffusion_model = any("diffusion_model" in key for key in key_list)

        if key_count < 1200 and has_input_blocks and has_diffusion_model:
            return "sd15"
        if key_count > 1200:
            return "sdxl"
        return None

    @staticmethod
    def load_lora(lora: LoraSpec) -> dict[str, torch.Tensor]:
        if lora.cached_lora is None:
            target_path = os.path.join(lora.subfolder, lora.name)
            lora_path = folder_paths.get_full_path("loras", target_path)

            if not lora_path:
                raise FileNotFoundError(f"LoRA '{target_path}' not found.")

            loaded_data, metadata = comfy.utils.load_torch_file(
                lora_path, return_metadata=True
            )
            lora.cached_lora = loaded_data
            lora.cached_metadata = metadata if metadata is not None else None

        return lora.cached_lora

    @staticmethod
    def get_model_key_set(model: comfy.model_patcher.ModelPatcher) -> set[str]:
        return set(model.model.state_dict().keys())

    @staticmethod
    def is_lora_compatible(
        lora_weights: dict[str, Any],
        model_or_keys: comfy.model_patcher.ModelPatcher | set[str] | None,
        lora: LoraSpec,
        model_spec: Any | None = None,
        clip: comfy.sd.CLIP | None = None,
    ) -> bool:
        if not lora_weights or model_or_keys is None:
            return False

        converted_lora = dict(comfy.lora_convert.convert_lora(dict(lora_weights)))
        if not converted_lora:
            return False

        model_profile = LoraProcessor._profile_for_model(model_or_keys, model_spec)
        lora_profile = LoraProcessor._profile_for_lora(converted_lora, lora)

        if not LoraProcessor._profiles_are_compatible(model_profile, lora_profile):
            return False

        if isinstance(model_or_keys, comfy.model_patcher.ModelPatcher):
            key_map = LoraProcessor._build_model_key_map(model_or_keys, clip)
            loaded = comfy.lora.load_lora(converted_lora, key_map, log_missing=False)
            if not loaded:
                return False

            lora_prefixes: set[str] = set()
            for k in converted_lora:
                for pat in ("lora_te2_", "lora_te1_", "lora_te_", "lora_unet_", "lora_prior_te_", "lora_prior_unet_"):
                    if k.startswith(pat):
                        lora_prefixes.add(pat)
                        break

            for pat in lora_prefixes:
                if not any(k.startswith(pat) for k in key_map):
                    return False

            return True

        if not LoraProcessor._has_lora_structure(converted_lora):
            return False

        return True

    @staticmethod
    def apply_lora(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP,
        loras: list[Any],
        model_spec: Any | None = None,
    ) -> tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        if not loras:
            return (model, clip)

        patched_model = model
        patched_clip = clip

        for lora in loras:
            try:
                lora_weights: dict[str, Any] = LoraProcessor.load_lora(lora)

                if not LoraProcessor.is_lora_compatible(
                    lora_weights,
                    patched_model,
                    lora,
                    model_spec=model_spec,
                    clip=patched_clip,
                ):
                    logger.warning(
                        "Skipping incompatible LoRA '%s' for model '%s'",
                        lora.name,
                        getattr(model_spec, "subfolder", "") or getattr(
                            model_spec, "name", "unknown"
                        ),
                    )
                    continue

                patched_model, patched_clip = LoraProcessor._apply_bypass_lora(
                    patched_model,
                    patched_clip,
                    lora_weights,
                    lora.weight,
                    lora.clip_weight,
                )
            except Exception as e:
                logger.error(f"Failed to load {lora.name}: {e}")
                continue

        return (patched_model, patched_clip)

    @staticmethod
    def _apply_bypass_lora(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP,
        lora_weights: dict[str, Any],
        strength_model: float,
        strength_clip: float,
    ) -> tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        key_map: dict[str, str] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        lora = comfy.lora_convert.convert_lora(lora_weights)
        loaded = comfy.lora.load_lora(lora, key_map)

        bypass_patches: dict[str, Any] = {}
        regular_patches: dict[str, Any] = {}
        for key, patch_data in loaded.items():
            if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                bypass_patches[key] = patch_data
            else:
                regular_patches[key] = patch_data

        new_modelpatcher = model.clone() if model is not None else None
        if new_modelpatcher is not None:
            if regular_patches:
                new_modelpatcher.add_patches(regular_patches, strength_model)

            if bypass_patches:
                model_sd_keys = set(new_modelpatcher.model.state_dict().keys())
                manager = comfy.weight_adapter.BypassInjectionManager()
                for key, adapter in bypass_patches.items():
                    if key in model_sd_keys:
                        manager.add_adapter(key, adapter, strength=strength_model)
                injections = manager.create_injections(new_modelpatcher.model)
                if manager.get_hook_count() > 0:
                    new_modelpatcher.set_injections("bypass_lora", injections)

        new_clip = clip.clone() if clip is not None else None
        if new_clip is not None:
            if regular_patches:
                new_clip.add_patches(regular_patches, strength_clip)

            if bypass_patches:
                clip_sd_keys = set(new_clip.cond_stage_model.state_dict().keys())
                clip_manager = comfy.weight_adapter.BypassInjectionManager()
                for key, adapter in bypass_patches.items():
                    if key in clip_sd_keys:
                        clip_manager.add_adapter(key, adapter, strength=strength_clip)
                clip_injections = clip_manager.create_injections(new_clip.cond_stage_model)
                if clip_manager.get_hook_count() > 0:
                    new_clip.patcher.set_injections("bypass_lora", clip_injections)

        return (new_modelpatcher, new_clip)



