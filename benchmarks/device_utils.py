"""Device resolution utilities shared across all PyPIELM benchmark scripts.

Key constraints:
- MPS (Apple Silicon): supports only float32; many linalg ops fall back to CPU
  unless ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set in the environment.
- CUDA: supports both float32 and float64; all linalg ops are native.
- CPU: supports float32 and float64; float64 is the default for PDE accuracy.

Usage::

    from device_utils import resolve_device, dtype_for_device, enable_mps_fallback

    device = resolve_device("mps")       # raises ValueError if unavailable
    dtype  = dtype_for_device(device)    # torch.float32 for MPS, float64 otherwise
    enable_mps_fallback()                # sets PYTORCH_ENABLE_MPS_FALLBACK=1
"""

from __future__ import annotations

import os
import sys
from typing import Union

import torch


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def enable_mps_fallback() -> None:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so linalg ops fall back to CPU.

    Must be called *before* any MPS tensor is created.  Calling it after the
    first MPS operation has no effect on already-dispatched kernels, but is
    harmless.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str) -> torch.device:
    """Validate and return a :class:`torch.device`.

    Raises:
        ValueError: if the requested device is not available on this machine.
    """
    requested = requested.lower().strip()

    if requested in ("cpu",):
        return torch.device("cpu")

    if requested in ("mps", "mps:0"):
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS device is not available on this machine. "
                "Run on Apple Silicon macOS 12.3+ with PyTorch ≥ 2.0."
            )
        enable_mps_fallback()
        return torch.device("mps")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device is not available on this machine. "
                "Ensure an NVIDIA GPU is present and CUDA drivers are installed."
            )
        return torch.device(requested)

    raise ValueError(
        f"Unknown device '{requested}'. "
        "Use 'cpu', 'mps', 'cuda', or 'cuda:N'."
    )


def dtype_for_device(device: Union[torch.device, str]) -> torch.dtype:
    """Return the appropriate default dtype for *device*.

    MPS does not support float64, so float32 is used there.  All other
    devices default to float64 for maximum PDE accuracy.
    """
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "mps":
        return torch.float32
    return torch.float64


def device_info(device: torch.device) -> dict:
    """Return a metadata dict describing the device (for JSON artifacts)."""
    info: dict = {"device": str(device), "dtype": str(dtype_for_device(device))}
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        info["mps_fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
    return info


def available_devices() -> list[str]:
    """Return a list of device strings available on the current machine."""
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devs.append(f"cuda:{i}")
    return devs


def to_device_dataset(dataset, device: torch.device) -> object:
    """Move a :class:`~pypielm.data.dataset.PIELMDataset` to *device*.

    Automatically casts to the appropriate dtype for the target device
    (float32 for MPS, float64 otherwise).
    """
    dtype = dtype_for_device(device)
    return dataset.to(str(device), dtype=dtype)


def to_device_tensor(arr, device: torch.device) -> torch.Tensor:
    """Convert a numpy array or tensor to the correct dtype/device."""
    import numpy as np
    dtype = dtype_for_device(device)
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, dtype=dtype, device=device)
    return arr.to(device=device, dtype=dtype)
