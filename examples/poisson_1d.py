"""Example: solve 1D Poisson equation with CorePIELM.

Problem
-------
    -u''(x) = π² sin(πx),   x ∈ [0, 1]
    u(0) = u(1) = 0

Exact solution: u(x) = sin(πx)

Run::

    cd PyPIELM
    python examples/poisson_1d.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pypielm.data.dataset import PIELMDataset
from pypielm.models import CorePIELM
from pypielm.core.solver import WeightedLinearSystem
from pypielm.visualization import plot_solution_1d, save_figure

# ---------------------------------------------------------------------------
# 1. Problem setup
# ---------------------------------------------------------------------------

N_COLLOC = 200     # interior collocation points
N_BC     = 2       # boundary points (just the two endpoints)
N_TEST   = 300     # dense evaluation grid for plotting
HIDDEN   = 300     # random neurons
SEED     = 42

rng = np.random.default_rng(SEED)

# Collocation points (interior)
X_c = rng.uniform(0.0, 1.0, (N_COLLOC, 1))

# Boundary conditions: u(0) = 0, u(1) = 0
X_bc = np.array([[0.0], [1.0]])
y_bc = np.array([0.0, 0.0])

dataset = PIELMDataset.from_arrays(
    X_c,
    X_bc=X_bc, y_bc=y_bc,
)

# ---------------------------------------------------------------------------
# 2. PDE operator  –u'' = π² sin(πx)  →  H'' = -π² sin(πx)
# ---------------------------------------------------------------------------

def pde_operator(fm, X: torch.Tensor) -> WeightedLinearSystem:
    """Encode  -u'' = π² sin(πx)  as a weighted linear block."""
    H_xx = fm.d2(X, 0)
    rhs = (math.pi ** 2
           * torch.sin(math.pi * X[:, 0:1]).to(X.dtype))
    # -u'' = π² sin  ⟺  -H'' β = rhs  ⟺  H_pde = -H_xx,  y = rhs
    return WeightedLinearSystem(H=-H_xx, y=rhs, weight=1.0)

# ---------------------------------------------------------------------------
# 3. Fit model
# ---------------------------------------------------------------------------

model = CorePIELM(hidden_dim=HIDDEN, ridge_lambda=1e-10, seed=SEED)
model.fit(dataset, pde_operator=pde_operator)

# ---------------------------------------------------------------------------
# 4. Evaluate on a dense test grid
# ---------------------------------------------------------------------------

X_test = np.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
u_true = np.sin(math.pi * X_test.squeeze())

with torch.no_grad():
    u_pred = model.predict(torch.tensor(X_test, dtype=torch.float64))
    u_pred_np = u_pred.numpy().squeeze()

rel_l2 = float(
    np.linalg.norm(u_true - u_pred_np) / (np.linalg.norm(u_true) + 1e-12)
)
print(f"Relative L² error: {rel_l2:.2e}")

# ---------------------------------------------------------------------------
# 5. Plot and save
# ---------------------------------------------------------------------------

fig = plot_solution_1d(
    X_test.squeeze(), u_pred_np, u_true,
    title="1D Poisson: CorePIELM solution",
    xlabel="x", ylabel="u(x)",
)
save_figure(fig, "poisson_1d_solution.png", dpi=200)
print("Figure saved → poisson_1d_solution.png")
