"""Model export to portable inference formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pypielm.core.base import BasePIELM


def _make_example_input(
    model: "BasePIELM",
    example_input: torch.Tensor | None,
    input_dim: int,
) -> torch.Tensor:
    if example_input is not None:
        return example_input
    # Try to infer input_dim from the feature map
    fm = getattr(model, "_fm", None)
    if fm is not None:
        input_dim = getattr(fm, "input_dim", input_dim)
    # Use the model's dtype for the dummy input
    dtype = getattr(model, "dtype", torch.float64)
    return torch.zeros(1, input_dim, dtype=dtype)


def to_onnx(
    model: "BasePIELM",
    path: str | Path,
    *,
    example_input: torch.Tensor | None = None,
    input_dim: int = 2,
    opset_version: int = 17,
) -> None:
    """Export *model* to ONNX format.

    Requires ``onnx`` and ``onnxruntime`` (install via ``pip install pypielm[export]``).

    Args:
        model: A fitted PIELM model.
        path: Destination ``.onnx`` file path.
        example_input: Representative input tensor ``(1, d)`` for tracing.
            If ``None``, a zero tensor of shape ``(1, input_dim)`` is used.
        input_dim: Spatial dimension *d* (used when ``example_input`` is None).
        opset_version: ONNX opset version.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ONNX export requires 'onnx'. Install with: pip install onnx onnxruntime"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = _make_example_input(model, example_input, input_dim)
    model.eval()

    torch.onnx.export(
        model,
        x,
        str(path),
        opset_version=opset_version,
        input_names=["X"],
        output_names=["u"],
        dynamic_axes={"X": {0: "batch_size"}, "u": {0: "batch_size"}},
    )


def to_torchscript(
    model: "BasePIELM",
    path: str | Path,
    *,
    example_input: torch.Tensor | None = None,
    input_dim: int = 2,
    method: str = "trace",
) -> torch.jit.ScriptModule:
    """Export *model* to TorchScript.

    Args:
        model: A fitted PIELM model.
        path: Destination ``.pt`` file path.
        example_input: Representative input for tracing.
        input_dim: Spatial dimension (used when ``example_input`` is None).
        method: ``'trace'`` or ``'script'``.

    Returns:
        The compiled :class:`torch.jit.ScriptModule`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = _make_example_input(model, example_input, input_dim)
    model.eval()

    if method == "trace":
        scripted = torch.jit.trace(model, x)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"method must be 'trace' or 'script', got '{method}'")

    torch.jit.save(scripted, str(path))
    return scripted

