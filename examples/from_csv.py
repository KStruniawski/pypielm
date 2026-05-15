"""Example: fit CorePIELM from a CSV file of observed (x, u) pairs.

Usage::

    cd PyPIELM
    # Generate a sample CSV then fit:
    python examples/from_csv.py

    # Use your own CSV (must have columns 'x' and 'u'):
    python examples/from_csv.py --data path/to/data.csv
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from pypielm.data import auto_load
from pypielm.models import CorePIELM
from pypielm.core.solver import WeightedLinearSystem
from pypielm.metrics.metrics import MetricsBundle


def _generate_sample_csv(path: Path, n: int = 150, seed: int = 0) -> None:
    """Write a synthetic noisy CSV with columns x,u."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    u = np.sin(math.pi * x) + 0.02 * rng.standard_normal(n)
    np.savetxt(
        path,
        np.column_stack([x, u]),
        delimiter=",",
        header="x,u",
        comments="",
    )
    print(f"Generated sample CSV → {path}  ({n} rows)")


def poisson_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    """Encode -u'' = π² sin(πx) (matches the ground truth)."""
    H_xx = fm.d2(X, 0)
    rhs  = (math.pi**2) * torch.sin(math.pi * X)
    return WeightedLinearSystem(-H_xx, rhs, weight=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit CorePIELM from CSV.")
    parser.add_argument(
        "--data", default=None,
        help="Path to CSV with columns 'x' and 'u'.  "
             "If omitted, a synthetic CSV is generated.",
    )
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if args.data is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        csv_path = Path(tmp.name)
        tmp.close()
        _generate_sample_csv(csv_path)
    else:
        csv_path = Path(args.data)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

    # --- Load via auto_load (CSVAdapter) ---
    ds = auto_load(
        csv_path,
        feature_cols=["x"],
        target_col="u",
        dtype=torch.float64,
    )
    print(f"Dataset: X_data={ds.X_data.shape}, y_data={ds.y_data.shape}")

    # --- Fit with physics regularisation ---
    import numpy as np
    from pypielm.data.dataset import PIELMDataset

    # Build collocation points for PDE residual
    rng = np.random.default_rng(args.seed)
    X_c = rng.uniform(0.0, 1.0, (300, 1))
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.array([[0.0], [0.0]])
    ds_full = PIELMDataset.from_arrays(
        X_c,
        X_data=ds.X_data.numpy(), y_data=ds.y_data.numpy(),
        X_bc=X_bc, y_bc=y_bc,
        dtype=torch.float64,
    )

    model = CorePIELM(hidden_dim=args.hidden, ridge_lambda=1e-8, seed=args.seed)
    model.fit(ds_full, pde_operator=poisson_op)

    # --- Evaluate on dense grid ---
    X_test  = torch.linspace(0.0, 1.0, 300).unsqueeze(1).double()
    u_pred  = model.predict(X_test)
    u_exact = torch.sin(math.pi * X_test)

    bundle = MetricsBundle(u_pred, u_exact)
    print(f"Relative L² error vs sin(πx): {bundle.rel_l2:.2e}")
    print(f"RMSE:                          {bundle.rmse:.2e}")


if __name__ == "__main__":
    main()
