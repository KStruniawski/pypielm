"""Example: solve 2D heat equation  u_t = Δu  on [0,1]² × [0,T].

Problem
-------
    u_t = u_xx + u_yy,   (x,y) ∈ [0,1]², t ∈ [0, T]
    u(0,y,t) = u(1,y,t) = u(x,0,t) = u(x,1,t) = 0  (Dirichlet)
    u(x,y,0) = sin(πx) sin(πy)                       (IC)

Exact solution: u(x,y,t) = exp(-2π²t) sin(πx) sin(πy)

Run::

    cd PyPIELM
    python examples/heat_2d.py
"""

from __future__ import annotations

import math
import numpy as np
import torch

from pypielm.data.dataset import PIELMDataset
from pypielm.core.solver import WeightedLinearSystem
from pypielm.models import CorePIELM
from pypielm.metrics.metrics import MetricsBundle

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

T       = 0.1
N_INT   = 2000   # interior (x, y, t) triples
N_BC    = 200    # points per spatial boundary edge
N_IC    = 400    # initial-condition points
N_TEST  = 20     # grid size per spatial axis at evaluation
HIDDEN  = 500
SEED    = 42
DTYPE   = torch.float64

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Build dataset
# ---------------------------------------------------------------------------

# Interior collocation: (x, y, t)
X_int = rng.uniform(0.0, 1.0, (N_INT, 2))
t_int = rng.uniform(0.0, T, (N_INT, 1))
Xc = np.hstack([X_int, t_int])  # (N_INT, 3)

# Boundary (zero Dirichlet on all four walls, t ∈ [0, T])
def _bc_edge(fixed_axis: int, fixed_val: float, n: int) -> np.ndarray:
    free = rng.uniform(0.0, 1.0, n)
    t_b  = rng.uniform(0.0, T, n)
    if fixed_axis == 0:   # x fixed
        return np.column_stack([np.full(n, fixed_val), free, t_b])
    else:                  # y fixed
        return np.column_stack([free, np.full(n, fixed_val), t_b])

X_bc = np.vstack([
    _bc_edge(0, 0.0, N_BC),
    _bc_edge(0, 1.0, N_BC),
    _bc_edge(1, 0.0, N_BC),
    _bc_edge(1, 1.0, N_BC),
])
y_bc = np.zeros((4 * N_BC, 1))

# Initial condition at t = 0
X_ic_xy = rng.uniform(0.0, 1.0, (N_IC, 2))
X_ic = np.column_stack([X_ic_xy, np.zeros(N_IC)])
y_ic = (np.sin(math.pi * X_ic_xy[:, 0])
        * np.sin(math.pi * X_ic_xy[:, 1]))[:, None]

dataset = PIELMDataset.from_arrays(
    Xc,
    X_bc=X_bc,  y_bc=y_bc,
    X_ic=X_ic,  y_ic=y_ic,
    dtype=DTYPE,
)

# ---------------------------------------------------------------------------
# 2. PDE operator: u_t - u_xx - u_yy = 0
# ---------------------------------------------------------------------------

def heat_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    H_t  = fm.d1(X, 2)   # ∂/∂t
    H_xx = fm.d2(X, 0)   # ∂²/∂x²
    H_yy = fm.d2(X, 1)   # ∂²/∂y²
    residual_H = H_t - H_xx - H_yy
    rhs = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
    return WeightedLinearSystem(residual_H, rhs, weight=1.0)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

model = CorePIELM(hidden_dim=HIDDEN, ridge_lambda=1e-8, seed=SEED, dtype=DTYPE)
model.fit(dataset, pde_operator=heat_op)

# ---------------------------------------------------------------------------
# 4. Evaluate on a structured grid at t = T
# ---------------------------------------------------------------------------

grid_1d = np.linspace(0.0, 1.0, N_TEST)
xx, yy  = np.meshgrid(grid_1d, grid_1d)
X_test  = np.column_stack([xx.ravel(), yy.ravel(), np.full(N_TEST**2, T)])

with torch.no_grad():
    u_pred = model.predict(torch.tensor(X_test, dtype=DTYPE)).numpy().ravel()

u_exact = (
    math.exp(-2 * math.pi**2 * T)
    * np.sin(math.pi * xx.ravel())
    * np.sin(math.pi * yy.ravel())
)

bundle = MetricsBundle(
    torch.tensor(u_pred),
    torch.tensor(u_exact),
)
print(f"Relative L² error at t={T}: {bundle.rel_l2:.2e}")
print(f"RMSE:                        {bundle.rmse:.2e}")
print(f"Max absolute error:          {bundle.max_err:.2e}")
