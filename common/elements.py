"""
All-to-Pipe element registry module.

Describes every field reachable inside a Pipe, so the nodes that touch one
element at a time (Export Single and Update Pipe) offer the same complete key
list instead of a hand-maintained subset.

A key is the dotted path of the field on the Pipe, for example
"parameters.steps". List entries carry their index, for example
"loras[0].weight"; only index 0 is offered, because the key list is built
before any Pipe exists.
"""

import copy
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ..alltopipe_types import (
    ImageConfig,
    NegativePrompt,
    Parameters,
    Pipe,
    PositivePrompt,
    Template,
    TemplateParser,
)
from .constants import (
    MAX_BATCH_SIZE,
    MAX_CFG,
    MAX_CLIP_SKIP,
    MAX_CLIP_WEIGHT,
    MAX_DECAY,
    MAX_DENOISE,
    MAX_HEIGHT,
    MAX_LORA_WEIGHT,
    MAX_NOISE,
    MAX_SEED,
    MAX_STEPS,
    MAX_WIDTH,
    MIN_BATCH_SIZE,
    MIN_CFG,
    MIN_CLIP_SKIP,
    MIN_CLIP_WEIGHT,
    MIN_DECAY,
    MIN_DENOISE,
    MIN_HEIGHT,
    MIN_LORA_WEIGHT,
    MIN_NOISE,
    MIN_SEED,
    MIN_STEPS,
    MIN_WIDTH,
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
)
from .validators import (
    validate_bool,
    validate_color_code,
    validate_float,
    validate_image,
    validate_int,
    validate_latent,
    validate_lora_list,
    validate_str,
)

logger: logging.Logger = logging.getLogger("AllToPipe")

# Fields of each Pipe container that are exposed as elements. What is left out
# is runtime state rather than something a user sets by hand: the caches on
# Model and LoraSpec, and the loaded ComfyUI objects handed out by export.
MODEL_FIELDS: tuple[str, ...] = ("name", "subfolder", "clip_skip")
LORA_FIELDS: tuple[str, ...] = ("name", "subfolder", "weight", "clip_weight")
PARAMETER_FIELDS: tuple[str, ...] = (
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "seed",
    "denoise",
)
IMAGE_SHAPING_FIELDS: tuple[str, ...] = (
    "width",
    "height",
    "batch_size",
    "noise",
    "color_code",
)
IMAGE_FIELDS: tuple[str, ...] = IMAGE_SHAPING_FIELDS + ("image", "latent")
TEMPLATE_FIELDS: tuple[str, ...] = (
    "text",
    "allow_parse_errors",
    "decay",
    "decay_floor",
)
PROMPT_FIELDS: tuple[str, ...] = ("allow_missing", "model", "lora")

# Used when a container has to be created because the field being written is
# not on the Pipe yet. They match the defaults the dedicated nodes (Parameters,
# Image Config) offer, so an update never invents a different one.
DEFAULT_PARAMETERS: dict[str, Any] = {
    "steps": 20,
    "cfg": 7.0,
    "sampler": "euler",
    "scheduler": "normal",
    "seed": 0,
    "denoise": 1.0,
}
DEFAULT_IMAGE_CONFIG: dict[str, Any] = {
    "width": 512,
    "height": 512,
    "batch_size": 1,
    "noise": 1.0,
    "color_code": None,
}


@dataclass(frozen=True)
class PipeElement:
    """
    One selectable field of a Pipe.

    path walks the Pipe to the field: attribute names for objects, integers
    for list entries. factory builds the container the field lives in when the
    Pipe does not have it yet, validate checks the value on its way in, and
    on_update repairs whatever the write invalidated.
    """

    key: str
    path: tuple[Any, ...]
    factory: Callable[[], Any] | None = None
    validate: Callable[[Any, str], Any] | None = None
    exportable: bool = True
    updateable: bool = True
    on_update: Callable[[Pipe, "PipeElement"], None] | None = None


def _numbers(minimum: float, maximum: float) -> Callable[[Any, str], Any]:
    """A FLOAT checker for a field, so ranges stay in one place."""
    return partial(validate_float, minimum=minimum, maximum=maximum)


def _whole_numbers(minimum: int, maximum: int) -> Callable[[Any, str], Any]:
    """An INT checker for a field, so ranges stay in one place."""
    return partial(validate_int, minimum=minimum, maximum=maximum)


def _required_text() -> Callable[[Any, str], Any]:
    """A STRING checker for a field that may not be blank."""
    return partial(validate_str, allow_empty=False)


# One checker per field, matching the ranges the dedicated nodes offer. A
# value that fails one is rejected before anything is written, so a bad edit
# can never reach the Pipe.
MODEL_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "name": _required_text(),
    "subfolder": validate_str,
    # ModelProcessor.load_model only accepts a negative layer index.
    "clip_skip": _whole_numbers(MIN_CLIP_SKIP, MAX_CLIP_SKIP),
}
LORA_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "name": _required_text(),
    "subfolder": validate_str,
    "weight": _numbers(MIN_LORA_WEIGHT, MAX_LORA_WEIGHT),
    "clip_weight": _numbers(MIN_CLIP_WEIGHT, MAX_CLIP_WEIGHT),
}
PARAMETER_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "steps": _whole_numbers(MIN_STEPS, MAX_STEPS),
    "cfg": _numbers(MIN_CFG, MAX_CFG),
    "sampler": partial(validate_str, choices=SUPPORTED_SAMPLERS, allow_empty=False),
    "scheduler": partial(validate_str, choices=SUPPORTED_SCHEDULERS, allow_empty=False),
    "seed": _whole_numbers(MIN_SEED, MAX_SEED),
    "denoise": _numbers(MIN_DENOISE, MAX_DENOISE),
}
IMAGE_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "width": _whole_numbers(MIN_WIDTH, MAX_WIDTH),
    "height": _whole_numbers(MIN_HEIGHT, MAX_HEIGHT),
    "batch_size": _whole_numbers(MIN_BATCH_SIZE, MAX_BATCH_SIZE),
    "noise": _numbers(MIN_NOISE, MAX_NOISE),
    "color_code": validate_color_code,
    "image": validate_image,
    "latent": validate_latent,
}
TEMPLATE_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    # Template rejects blank text, so writing one here would fail anyway.
    "text": _required_text(),
    "allow_parse_errors": validate_bool,
    "decay": _numbers(MIN_DECAY, MAX_DECAY),
    "decay_floor": _numbers(MIN_DECAY, MAX_DECAY),
}
PROMPT_VALIDATORS: dict[str, Callable[[Any, str], Any]] = {
    "allow_missing": validate_bool,
    "model": validate_str,
    "lora": validate_str,
    **dict.fromkeys(PositivePrompt.ALLOWED_FEATURES, validate_str),
}


def _make_template(kind: str) -> Callable[[], Template]:
    """
    Builds the Template that a written text needs to land on.

    Only the text key uses this: Template rejects blank text, so any other
    template field has nothing valid to fall back on and is refused instead.
    The stub is replaced by the write that follows it.
    """

    def factory() -> Template:
        return Template(kind, [], "placeholder", True, 0.0, 0.5)

    return factory


def _rebuild_template(pipe: Pipe, element: PipeElement) -> None:
    """A template field changed, so anything derived from it is stale."""
    template = getattr(pipe, element.path[0])
    template.parsed_template = None
    if element.path[-1] == "text":
        template.placeholders = TemplateParser.find_placeholders(template.text)


def _rebuild_prompt(pipe: Pipe, element: PipeElement) -> None:
    """A prompt field changed, so the template rendered from it is stale."""
    attribute = (
        "positive_template"
        if element.path[0] == "positive_prompt"
        else "negative_template"
    )
    template = getattr(pipe, attribute)
    if template is not None:
        template.parsed_template = None


def _rebuild_image(pipe: Pipe, element: PipeElement) -> None:
    """Keeps image_config consistent with the field that just changed."""
    config = pipe.image_config
    if config is None:
        return

    attribute = element.path[-1]
    if attribute == "image":
        _encode_image(config, pipe)
    elif attribute in IMAGE_SHAPING_FIELDS:
        config.image = None
        config.latent = None


def _encode_image(config: ImageConfig, pipe: Pipe) -> None:
    """Encodes the latent for a new image, when a VAE is already loaded."""
    if config.image is None:
        config.latent = None
        return

    cached = getattr(pipe.model, "cached_model", None)
    if cached is None:
        logger.warning(
            "No model is loaded, so no latent is encoded here. Export builds it."
        )
        config.latent = None
        return

    _, _, vae = cached
    config.latent = {"samples": vae.encode(config.image[:, :, :, :3])}


def _drop_model_cache(pipe: Pipe, element: PipeElement) -> None:
    """Pointing the Pipe at another checkpoint invalidates what was loaded."""
    if element.path[-1] in ("name", "subfolder"):
        if hasattr(pipe.model, "cached_model"):
            pipe.model.cached_model = None


def _drop_lora_cache(pipe: Pipe, element: PipeElement) -> None:
    """Changed LoRA weights mean the cached tensors no longer match the spec."""
    index = element.path[1] if len(element.path) > 1 else None
    specs = [pipe.loras[index]] if isinstance(index, int) else pipe.loras

    for spec in specs:
        spec.cached_lora = None
        spec.cached_metadata = None


def _build_elements() -> dict[str, PipeElement]:
    elements: dict[str, PipeElement] = {}

    def add(element: PipeElement) -> None:
        if element.key in elements:
            raise ValueError(f"Duplicate element key: {element.key}")
        elements[element.key] = element

    def add_fields(
        container: str,
        fields: tuple[str, ...],
        factory: Callable[[], Any] | None = None,
        factories: dict[str, Callable[[], Any]] | None = None,
        on_update: Callable[[Pipe, PipeElement], None] | None = None,
        validators: dict[str, Callable[[Any, str], Any]] | None = None,
    ) -> None:
        for name in fields:
            add(
                PipeElement(
                    key=f"{container}.{name}",
                    path=(container, name),
                    factory=factory or (factories or {}).get(name),
                    validate=(validators or {}).get(name),
                    on_update=on_update,
                )
            )

    # Model is not created on the fly: a nameless Model is not a usable model.
    add_fields(
        "model", MODEL_FIELDS, on_update=_drop_model_cache, validators=MODEL_VALIDATORS
    )

    add(
        PipeElement(
            key="loras",
            path=("loras",),
            validate=validate_lora_list,
            on_update=_drop_lora_cache,
        )
    )
    for name in LORA_FIELDS:
        add(
            PipeElement(
                key=f"loras[0].{name}",
                path=("loras", 0, name),
                validate=LORA_VALIDATORS[name],
                on_update=_drop_lora_cache,
            )
        )

    add_fields(
        "parameters",
        PARAMETER_FIELDS,
        factory=lambda: Parameters(**DEFAULT_PARAMETERS),
        validators=PARAMETER_VALIDATORS,
    )
    add_fields(
        "image_config",
        IMAGE_FIELDS,
        factory=lambda: ImageConfig(**DEFAULT_IMAGE_CONFIG),
        on_update=_rebuild_image,
        validators=IMAGE_VALIDATORS,
    )

    for container in ("positive_template", "negative_template"):
        add_fields(
            container,
            TEMPLATE_FIELDS,
            factories={"text": _make_template(container.removesuffix("_template"))},
            on_update=_rebuild_template,
            validators=TEMPLATE_VALIDATORS,
        )
        # Derived from the other template fields, so readable but never written.
        add(
            PipeElement(
                key=f"{container}.placeholders",
                path=(container, "placeholders"),
                updateable=False,
            )
        )
        add(
            PipeElement(
                key=f"{container}.parsed_template",
                path=(container, "parsed_template"),
                updateable=False,
            )
        )

    add_fields(
        "positive_prompt",
        PROMPT_FIELDS + PositivePrompt.ALLOWED_FEATURES,
        factory=PositivePrompt,
        on_update=_rebuild_prompt,
        validators=PROMPT_VALIDATORS,
    )
    add_fields(
        "negative_prompt",
        PROMPT_FIELDS + NegativePrompt.ALLOWED_FEATURES,
        factory=NegativePrompt,
        on_update=_rebuild_prompt,
        validators=PROMPT_VALIDATORS,
    )

    return elements


class ElementRegistry:
    """Lookup and traversal for the elements of a Pipe."""

    elements: dict[str, PipeElement] = _build_elements()

    @classmethod
    def get(cls, key: str) -> PipeElement:
        """
        Return the element a key names.

        Raises:
            ValueError: If the key is not part of a Pipe
        """
        element = cls.elements.get(key)
        if element is None:
            raise ValueError(f"'{key}' is not an element of a Pipe")
        return element

    @classmethod
    def export_keys(cls) -> tuple[str, ...]:
        """Keys that Export Single can read, in list order."""
        return tuple(key for key, element in cls.elements.items() if element.exportable)

    @classmethod
    def update_keys(cls) -> tuple[str, ...]:
        """Keys that Update Pipe can write, in list order."""
        return tuple(key for key, element in cls.elements.items() if element.updateable)

    @classmethod
    def read(cls, pipe: Pipe, key: str) -> Any:
        """
        Read one element out of a Pipe.

        Raises:
            ValueError: If the key is not readable, or the path is not set
        """
        element = cls.get(key)
        if not element.exportable:
            raise ValueError(f"'{key}' can be written but not read")

        value: Any = pipe
        for part in element.path:
            value = cls._descend(value, part, key)
        return value

    @classmethod
    def write(cls, pipe: Pipe, key: str, new_value: Any) -> Pipe:
        """
        Return a copy of the Pipe with one element replaced.

        The value is checked before anything is copied, so a rejected value
        leaves the Pipe it was handed exactly as it was.

        Raises:
            ValueError: If the key is not writable, the value fails its check,
                or the path cannot be built
        """
        element = cls.get(key)
        if not element.updateable:
            raise ValueError(f"'{key}' is derived from other fields and cannot be set")

        if element.validate is not None:
            new_value = element.validate(new_value, key)

        new_pipe: Pipe = pipe.clone()

        root: Any = element.path[0]
        if getattr(new_pipe, root, None) is None:
            if element.factory is None:
                raise ValueError(f"'{key}' cannot be set: the Pipe has no {root}")
            setattr(new_pipe, root, element.factory())

        # Pipe.clone() shares its containers with the Pipe it came from, so each
        # one on the way to the field is replaced by a copy first. The value
        # written at the end then lands on the copy alone.
        owner: Any = new_pipe
        for index, part in enumerate(element.path[:-1]):
            following: Any = element.path[index + 1]
            child: Any = cls._descend(owner, part, key)

            if isinstance(following, int):
                if not child:
                    raise ValueError(f"'{key}' cannot be set: the Pipe has no {root}")
                child = list(child)
            else:
                child = copy.copy(child)

            cls._assign(owner, part, child)
            owner = child

        cls._assign(owner, element.path[-1], new_value)

        if element.on_update is not None:
            element.on_update(new_pipe, element)

        return new_pipe

    @staticmethod
    def _descend(owner: Any, part: Any, key: str) -> Any:
        """Follow one step of an element path, raising when the step is missing."""
        if isinstance(part, int):
            if not isinstance(owner, list) or part >= len(owner):
                raise ValueError(f"'{key}' cannot be resolved: entry {part} is missing")
            return owner[part]

        value = getattr(owner, part, None)
        if value is None:
            raise ValueError(f"'{key}' cannot be resolved: '{part}' holds no value")
        return value

    @staticmethod
    def _assign(owner: Any, part: Any, value: Any) -> None:
        """Write one step of an element path, on lists and objects alike."""
        if isinstance(part, int):
            owner[part] = value
        else:
            setattr(owner, part, value)