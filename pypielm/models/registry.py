"""Model registry: maps string names to PIELM/PINN model classes.

The registry is populated automatically via the :func:`register` decorator.
All model classes in this package self-register at import time, so YAML configs
and CLI commands can reference models by name without hardcoded imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypielm.core.base import BasePIELM

# Global registry mapping lowercase model name → class
MODEL_REGISTRY: dict[str, type[BasePIELM]] = {}


def register(name: str):
    """Class decorator that registers a PIELM/PINN model under ``name``.

    Args:
        name: The string key used in YAML configs and the CLI.

    Returns:
        The unmodified class (decorator is side-effect only).

    Example::

        @register("vanilla_pielm")
        class VanillaPIELM(BasePIELM):
            ...

    """
    def _decorator(cls: type) -> type:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return _decorator


def get_model(name: str, **kwargs: Any) -> BasePIELM:
    """Instantiate a registered model by name.

    Args:
        name: Model name as registered via :func:`register`.  Case-insensitive.
        **kwargs: Constructor arguments forwarded to the model class.

    Returns:
        Instantiated model object.

    Raises:
        KeyError: If ``name`` is not in the registry.

    Example::

        model = get_model("core_pielm", hidden_dim=300, ridge_lambda=1e-8)

    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: [{available}]"
        )
    return MODEL_REGISTRY[key](**kwargs)
