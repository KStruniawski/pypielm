"""Model checkpointing: save and load trained PIELM weights."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pypielm.core.base import BasePIELM

_VERSION = "0.1.0"


def _get_model_registry_name(model: "BasePIELM") -> str:
    """Return the registry key for *model* (e.g. ``'core_pielm'``)."""
    from pypielm.models.registry import MODEL_REGISTRY
    cls = type(model)
    for name, registered_cls in MODEL_REGISTRY.items():
        if registered_cls is cls:
            return name
    return f"{cls.__module__}.{cls.__qualname__}"


def _extract_config(model: "BasePIELM") -> dict[str, Any]:
    """Extract constructor kwargs from a model's public attributes."""
    config: dict[str, Any] = {}
    for k, v in vars(model).items():
        if k.startswith("_"):
            continue
        # Skip non-serialisable torch types for JSON compatibility but keep dtype name
        if isinstance(v, torch.dtype):
            config[k] = str(v)
        elif isinstance(v, torch.device):
            config[k] = str(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            config[k] = v
    return config


def save_model(
    model: "BasePIELM",
    path: str | Path,
    *,
    include_config: bool = True,
    overwrite: bool = False,
) -> None:
    """Serialise *model* weights (and optionally config) to *path*.

    The checkpoint format is a ``torch.save``-compatible dict::

        {
            "version": "0.1.0",
            "model_class": "<registry name or qualified class name>",
            "state_dict": { ... },
            "config": { ... },   # only when include_config=True
        }

    Args:
        model: A fitted :class:`~pypielm.core.base.BasePIELM` instance.
        path: Destination file path (``.pt`` extension recommended).
        include_config: Whether to embed the model's config in the checkpoint.
        overwrite: If ``False``, raise :class:`FileExistsError` when *path*
            already exists.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {path}. Pass overwrite=True to replace.")

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": _VERSION,
        "model_class": _get_model_registry_name(model),
        "state_dict": model.state_dict(),
    }
    if include_config:
        payload["config"] = _extract_config(model)

    torch.save(payload, path)


def load_model(
    path: str | Path,
    *,
    model_class: "type[BasePIELM] | None" = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> "BasePIELM":
    """Load a checkpoint written by :func:`save_model`.

    If *model_class* is ``None``, the class is inferred from the checkpoint's
    ``model_class`` field via the model registry.

    Args:
        path: Path to the checkpoint file.
        model_class: Override the class to use for reconstruction.
        device: Device to load the model onto.
        dtype: Dtype override (default: use saved dtype).

    Returns:
        A :class:`~pypielm.core.base.BasePIELM` instance with weights loaded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    if model_class is None:
        class_name: str = payload["model_class"]
        from pypielm.models.registry import MODEL_REGISTRY
        if class_name in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY[class_name]
        else:
            # Try qualified import: "module.ClassName"
            parts = class_name.rsplit(".", 1)
            if len(parts) == 2:
                import importlib
                mod = importlib.import_module(parts[0])
                model_class = getattr(mod, parts[1])
            else:
                raise ValueError(f"Cannot resolve model class '{class_name}'. Pass model_class= explicitly.")

    config: dict[str, Any] = payload.get("config", {})

    # Apply device/dtype overrides
    device_str = str(device)
    if device_str:
        config["device"] = device_str
    if dtype is not None:
        config["dtype"] = dtype

    # Remove keys that are not constructor arguments (dtype stored as string)
    # Reconstruct dtype from string if needed
    if "dtype" in config and isinstance(config["dtype"], str):
        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.float16": torch.float16,
        }
        config["dtype"] = dtype_map.get(config["dtype"], torch.float64)

    # Instantiate model; pass only keys the constructor accepts
    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_keys = set(sig.parameters) - {"self"}
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    model = model_class(**filtered_config)

    # For PIELM models with lazy _fm init: if the state dict contains _fm.*
    # keys, we need to construct _fm before loading so it is registered.
    state = payload["state_dict"]
    _maybe_init_fm(model, state, filtered_config)

    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def _maybe_init_fm(model: Any, state: dict, config: dict) -> None:
    """Pre-register _fm if the state_dict contains its weights but model._fm is None."""
    fm_keys = [k for k in state if k.startswith("_fm.")]
    if not fm_keys:
        return

    fm = getattr(model, "_fm", None)
    if fm is None:
        # Infer input_dim from _fm.W shape: (input_dim, hidden_dim)
        w_key = next((k for k in fm_keys if k.endswith(".W")), None)
        if w_key is not None:
            input_dim = state[w_key].shape[0]
            hidden_dim_ckpt = state[w_key].shape[1]
            # Override hidden_dim if it was not in config or differs
            if hasattr(model, "hidden_dim") and model.hidden_dim != hidden_dim_ckpt:
                model.hidden_dim = hidden_dim_ckpt

            _build_fm = getattr(model, "_build_fm", None)
            if callable(_build_fm):
                model._fm = _build_fm(input_dim)

    # Register _beta buffer if not yet present
    if "_beta" in state:
        existing_beta = getattr(model, "_beta", None)
        if existing_beta is None:
            # Register with correct shape so load_state_dict can match it
            model.register_buffer("_beta", torch.zeros_like(state["_beta"]))



