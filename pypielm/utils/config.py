"""YAML-based experiment configuration loader and runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Fully-specified configuration for a single PyPIELM experiment.

    All fields can be populated from a YAML file via :func:`load_config`.

    Args:
        model: Registered model name (e.g. ``'core_pielm'``).
        model_kwargs: Keyword arguments forwarded to the model constructor.
        data: Data loading specification (``source``, ``path``, split ratios).
        pde: PDE configuration (``operator``, ``collocation``, ``n_collocation``).
        seed: Global random seed (passed to :func:`~pypielm.utils.seed_everything`).
        device: Target device string.
        output_dir: Directory for saving artefacts (model, results, figures).

    Example YAML::

        model: core_pielm
        model_kwargs:
          hidden_dim: 300
          ridge_lambda: 1.0e-8
        data:
          source: pinnacle
          path: data/poisson_classic.dat
          val_ratio: 0.1
          test_ratio: 0.2
        pde:
          operator: laplacian
          collocation: LHSSampler
          n_collocation: 1000
        seed: 42
        device: cpu
        output_dir: runs/poisson_classic_core/
    """

    model: str = "core_pielm"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    pde: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "runs/"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_DEVICES = {"cpu", "cuda", "mps"}
_VALID_SAMPLERS = {"UniformSampler", "LHSSampler", "AdaptiveSampler", "GridSampler", None}


def _validate_config(cfg: ExperimentConfig) -> None:
    """Raise :class:`ValueError` for clearly invalid config fields."""
    from pypielm.models.registry import MODEL_REGISTRY  # populated at import

    if not cfg.model:
        raise ValueError("'model' must be a non-empty string.")

    key = cfg.model.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Model '{cfg.model}' not found in registry. "
            f"Available: [{available}]"
        )

    dev = cfg.device.lower()
    if not (dev in _VALID_DEVICES or dev.startswith("cuda:")):
        raise ValueError(
            f"'device' must be one of {sorted(_VALID_DEVICES)} or 'cuda:N', "
            f"got '{cfg.device}'."
        )

    if not isinstance(cfg.seed, int):
        raise ValueError(f"'seed' must be an integer, got {type(cfg.seed).__name__}.")

    data = cfg.data
    if "path" in data and not Path(str(data["path"])).exists():
        raise ValueError(f"Data path does not exist: {data['path']}")

    pde = cfg.pde
    if "collocation" in pde and pde["collocation"] not in _VALID_SAMPLERS:
        raise ValueError(
            f"Unknown collocation sampler '{pde['collocation']}'. "
            f"Valid: {sorted(s for s in _VALID_SAMPLERS if s)}."
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the ``.yaml`` configuration file.

    Returns:
        A populated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required fields are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = ExperimentConfig(
        model=str(raw.get("model", "core_pielm")),
        model_kwargs=dict(raw.get("model_kwargs", {})),
        data=dict(raw.get("data", {})),
        pde=dict(raw.get("pde", {})),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cpu")),
        output_dir=str(raw.get("output_dir", "runs/")),
    )

    _validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# PDE operator resolver
# ---------------------------------------------------------------------------

def _resolve_pde_operator(pde: dict[str, Any]) -> Any:
    """Instantiate a PDE operator object from the ``pde`` config block.

    For function-style operators (``laplacian``, ``gradient``, etc.) ``None``
    is returned; the model's ``fit`` method receives them via separate keyword
    arguments when needed.  ``analytic_laplacian`` returns an instantiated
    :class:`~pypielm.pde.operators.AnalyticLaplacian`.
    """
    op_name = pde.get("operator")
    if op_name is None:
        return None

    op_lower = op_name.lower()
    if op_lower in {"laplacian", "gradient", "divergence", "advection_term"}:
        return None  # function-style; model uses them internally

    if op_lower in {"analytic_laplacian", "analyticlaplacian"}:
        from pypielm.pde.operators import AnalyticLaplacian
        return AnalyticLaplacian()

    raise ValueError(
        f"Unknown pde.operator '{op_name}'. "
        "Supported: 'laplacian', 'gradient', 'divergence', "
        "'advection_term', 'analytic_laplacian'."
    )


# ---------------------------------------------------------------------------
# Collocation sampler resolver
# ---------------------------------------------------------------------------

def _resolve_sampler(pde: dict[str, Any]) -> Any:
    """Instantiate a collocation sampler from the ``pde`` config block."""
    name = pde.get("collocation")
    if name is None:
        return None

    n = int(pde.get("n_collocation", 500))
    lb = pde.get("domain_lb", [0.0])
    ub = pde.get("domain_ub", [1.0])
    seed = pde.get("seed", 42)

    from pypielm.pde.collocation import (
        BoxDomain,
        GridSampler,
        LHSSampler,
        UniformSampler,
    )

    domain = BoxDomain(lb=lb, ub=ub)

    if name == "UniformSampler":
        return UniformSampler(domain, n_points=n, seed=seed)
    if name == "LHSSampler":
        return LHSSampler(domain, n_points=n, seed=seed)
    if name == "GridSampler":
        nx = pde.get("nx", n)
        ny = pde.get("ny")
        kw: dict[str, Any] = {"nx": nx}
        if ny is not None:
            kw["ny"] = ny
        return GridSampler(domain, **kw)
    if name == "AdaptiveSampler":
        # Requires a residual_fn at runtime; return None and let model decide.
        return None

    raise ValueError(f"Unknown collocation sampler '{name}'.")


# ---------------------------------------------------------------------------
# Dataset loader helper
# ---------------------------------------------------------------------------

def _load_dataset(config: ExperimentConfig) -> Any:
    """Load a :class:`~pypielm.data.PIELMDataset` from the data block."""
    import math

    import torch

    from pypielm.data import auto_load
    from pypielm.data.dataset import PIELMDataset

    data = config.data
    path = data.get("path")

    if path is not None:
        kw: dict[str, Any] = {k: v for k, v in data.items()
                               if k not in ("path", "source")}
        return auto_load(path, device=config.device, **kw)

    # No path → build trivial synthetic 1-D sinusoidal dataset for dry-runs.
    # X_colloc are interior domain points; X_data/y_data are the observed data
    # the model will regress on (same points for simplicity).
    n = int(data.get("n_samples", 200))
    noise = float(data.get("noise", 0.0))
    dtype = torch.float64
    device = config.device
    X = torch.linspace(0.0, 1.0, n, dtype=dtype).unsqueeze(1).to(device)
    y = torch.sin(2 * math.pi * X)
    if noise > 0.0:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(config.seed)
        y = y + noise * torch.randn(X.shape, dtype=dtype, generator=rng).to(device)
    return PIELMDataset.from_arrays(
        X,
        X_data=X,
        y_data=y,
        dtype=dtype,
        device=device,
    )


# ---------------------------------------------------------------------------
# Artifact saver
# ---------------------------------------------------------------------------

def _save_artifacts(
    config: ExperimentConfig,
    model: Any,
    metrics: dict[str, Any],
) -> list[str]:
    """Save checkpoint and ``results.json``; return list of written paths."""
    from pypielm.io.checkpoint import save_model

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []

    # Model checkpoint
    ckpt_path = out_dir / "model.pt"
    try:
        save_model(model, ckpt_path, overwrite=True)
        artifacts.append(str(ckpt_path))
    except Exception:
        pass  # some models may not support full state_dict yet

    # Results JSON
    results = {
        "metrics": {
            k: (v if not hasattr(v, "item") else v.item())
            for k, v in metrics.items()
        },
        "config": {
            "model": config.model,
            "model_kwargs": config.model_kwargs,
            "seed": config.seed,
            "device": config.device,
            "output_dir": config.output_dir,
        },
    }
    results_path = out_dir / "results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    artifacts.append(str(results_path))

    return artifacts


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute a single experiment defined by ``config``.

    Steps performed:

    1. Seed everything via :func:`~pypielm.utils.seed_everything`.
    2. Load data via :func:`~pypielm.data.auto_load` (if ``data.path`` is
       given) or build a synthetic dataset for dry-runs.
    3. Instantiate the model from the registry.
    4. Resolve PDE operator and collocation sampler.
    5. Call ``model.fit(dataset, ...)``.
    6. Evaluate metrics on the test split.
    7. Save checkpoint + ``results.json`` to ``output_dir``.

    Args:
        config: A validated :class:`ExperimentConfig`.

    Returns:
        Dictionary with keys:

        * ``'metrics'``: dict of metric name → float.
        * ``'config'``: the config as a plain dict.
        * ``'artifacts'``: list of paths of saved files.
    """
    import torch

    from pypielm.metrics.metrics import MetricsBundle
    from pypielm.models.registry import get_model
    from pypielm.utils.reproducibility import seed_everything

    seed_everything(config.seed)

    dataset = _load_dataset(config)

    model_kw: dict[str, Any] = dict(config.model_kwargs)
    model_kw.setdefault("device", config.device)
    model_kw.setdefault("seed", config.seed)
    model = get_model(config.model, **model_kw)

    pde_operator = _resolve_pde_operator(config.pde)
    sampler = _resolve_sampler(config.pde)

    t0 = time.perf_counter()
    model.fit(dataset, pde_operator=pde_operator, collocation_sampler=sampler)
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        X_eval = dataset.X_data if dataset.X_data is not None else dataset.X_colloc
        y_eval = dataset.y_data

    if y_eval is not None:
        with torch.no_grad():
            y_pred = model.predict(X_eval)
        bundle = MetricsBundle(y_pred, y_eval)
        metrics: dict[str, Any] = bundle.to_dict()
    else:
        metrics = {}
    metrics["fit_time_s"] = fit_time

    artifacts = _save_artifacts(config, model, metrics)

    return {
        "metrics": metrics,
        "config": {
            "model": config.model,
            "model_kwargs": config.model_kwargs,
            "data": config.data,
            "pde": config.pde,
            "seed": config.seed,
            "device": config.device,
            "output_dir": config.output_dir,
        },
        "artifacts": artifacts,
    }
