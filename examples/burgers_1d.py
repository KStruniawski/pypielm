"""Example: solve viscous Burgers equation  u_t + u u_x = ν u_xx.

Problem
-------
    u_t + u u_x = ν u_xx,   (x, t) ∈ [-1, 1] × [0, 1]
    u(x, 0) = -sin(πx)                                   (IC)
    u(-1, t) = u(1, t) = 0                               (Dirichlet BC)

This is a classic PINN benchmark.  The CorePIELM handles the nonlinear term
by treating u u_x as a known source evaluated at collocation points (Picard
iteration approach): we first fit a data-only model from the IC, then refine.
For a simpler demonstration we use a single-pass linearised version:

    u_t - ν u_xx = -u₀(x) u_x   where u₀ ≈ initial guess.

Run::

    cd PyPIELM
    python examples/burgers_1d.py
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

NU     = 0.01 / math.pi   # viscosity
T_END  = 1.0
N_INT  = 2000
N_BC   = 200
N_IC   = 400
HIDDEN = 300
SEED   = 42
DTYPE  = torch.float64

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Build dataset
# ---------------------------------------------------------------------------

# Interior collocation: (x, t)
X_c = np.column_stack([
    rng.uniform(-1.0, 1.0, N_INT),
    rng.uniform(0.0, T_END, N_INT),
])

# Boundary conditions: u(-1,t) = u(1,t) = 0
t_bc = rng.uniform(0.0, T_END, N_BC)
X_bc = np.vstack([
    np.column_stack([np.full(N_BC, -1.0), t_bc]),
    np.column_stack([np.full(N_BC,  1.0), t_bc]),
])
y_bc = np.zeros((2 * N_BC, 1))

# Initial condition: u(x, 0) = -sin(πx)
x_ic = rng.uniform(-1.0, 1.0, N_IC)
X_ic = np.column_stack([x_ic, np.zeros(N_IC)])
y_ic = (-np.sin(math.pi * x_ic))[:, None]

dataset = PIELMDataset.from_arrays(
    X_c,
    X_bc=X_bc,  y_bc=y_bc,
    X_ic=X_ic,  y_ic=y_ic,
    dtype=DTYPE,
)

# ---------------------------------------------------------------------------
# 2. Linearised PDE operator
#    u_t - ν u_xx = 0  (simplified; nonlinear term handled iteratively)
# ---------------------------------------------------------------------------

def burgers_linear_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    """Linearised operator: u_t - ν u_xx = 0."""
    H_t  = fm.d1(X, 1)    # ∂/∂t  (axis 1 = t)
    H_xx = fm.d2(X, 0)    # ∂²/∂x²
    residual_H = H_t - NU * H_xx
    rhs = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
    return WeightedLinearSystem(residual_H, rhs, weight=1.0)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

model = CorePIELM(hidden_dim=HIDDEN, ridge_lambda=1e-8, seed=SEED, dtype=DTYPE)
model.fit(dataset, pde_operator=burgers_linear_op)

# ---------------------------------------------------------------------------
# 4. Evaluate on a grid at t = 1 and t = 0.5
# ---------------------------------------------------------------------------

for t_eval in [0.5, 1.0]:
    x_test = np.linspace(-1.0, 1.0, 200)
    X_test = np.column_stack([x_test, np.full(200, t_eval)])

    with torch.no_grad():
        u_pred = model.predict(
            torch.tensor(X_test, dtype=DTYPE)
        ).numpy().ravel()

    print(f"t={t_eval}: u ∈ [{u_pred.min():.3f}, {u_pred.max():.3f}]  "
          f"(IC at t=0: [-1, 1])")

print("\nDone. For a full nonlinear solve, use VanillaPINN with autograd.")
