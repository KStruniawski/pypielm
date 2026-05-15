# Tutorial: 2D Heat Equation

This tutorial solves the 2D heat equation

$$u_t = \Delta u, \quad (x, y) \in [0,1]^2, \quad t \in [0, T]$$

with a known exact solution for error measurement.

## Problem setup

We treat time as an additional spatial coordinate: the input is $(x, y, t)$
and the PDE residual is $u_t - u_{xx} - u_{yy} = 0$.

The manufactured exact solution is:

$$u(x, y, t) = e^{-2\pi^2 t} \sin(\pi x) \sin(\pi y)$$

satisfying zero Dirichlet BCs on all spatial boundaries and the IC
$u_0(x, y) = \sin(\pi x)\sin(\pi y)$.

## Dataset

```python
import math
import numpy as np
from pypielm.data.dataset import PIELMDataset

rng = np.random.default_rng(42)
T = 0.1

# Interior: (x, y, t) triples
N_int = 2000
X_int = rng.uniform(0.0, 1.0, (N_int, 2))  # (x, y)
t_int = rng.uniform(0.0, T, (N_int, 1))
Xc = np.hstack([X_int, t_int])  # (N, 3)

# Boundary: zero Dirichlet on x=0,1 and y=0,1
N_bc = 200
t_bc = rng.uniform(0.0, T, (4 * N_bc, 1))
x_bc = np.vstack([
    np.column_stack([np.zeros(N_bc), rng.uniform(0, 1, N_bc), t_bc[:N_bc, 0]]),
    np.column_stack([np.ones(N_bc),  rng.uniform(0, 1, N_bc), t_bc[N_bc:2*N_bc, 0]]),
    np.column_stack([rng.uniform(0, 1, N_bc), np.zeros(N_bc), t_bc[2*N_bc:3*N_bc, 0]]),
    np.column_stack([rng.uniform(0, 1, N_bc), np.ones(N_bc),  t_bc[3*N_bc:, 0]]),
])
y_bc = np.zeros((4 * N_bc, 1))

# Initial condition at t=0
N_ic = 300
X_ic_xy = rng.uniform(0, 1, (N_ic, 2))
X_ic = np.column_stack([X_ic_xy, np.zeros(N_ic)])
y_ic = (np.sin(math.pi * X_ic_xy[:, 0]) * np.sin(math.pi * X_ic_xy[:, 1]))[:, None]

ds = PIELMDataset.from_arrays(Xc, X_bc=x_bc, y_bc=y_bc, X_ic=X_ic, y_ic=y_ic)
```

## PDE operator

```python
import torch
from pypielm.core.solver import WeightedLinearSystem

def heat_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    # Residual: u_t - u_xx - u_yy = 0
    H_t  = fm.d1(X, 2)   # ∂/∂t  (axis 2)
    H_xx = fm.d2(X, 0)   # ∂²/∂x²
    H_yy = fm.d2(X, 1)   # ∂²/∂y²
    residual_H = H_t - H_xx - H_yy
    rhs = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
    return WeightedLinearSystem(residual_H, rhs, weight=1.0)
```

## Training and evaluation

```python
from pypielm.models import CorePIELM

model = CorePIELM(hidden_dim=500, ridge_lambda=1e-8, seed=42)
model.fit(ds, pde_operator=heat_op)

# Test on a grid at t=T
xy = np.mgrid[0:1:30j, 0:1:30j].reshape(2, -1).T
X_test = np.column_stack([xy, np.full(len(xy), T)])
u_pred  = model.predict(torch.tensor(X_test, dtype=torch.float64)).numpy()
u_exact = (
    math.exp(-2 * math.pi**2 * T)
    * np.sin(math.pi * xy[:, 0])
    * np.sin(math.pi * xy[:, 1])
)
rel_l2 = np.linalg.norm(u_pred.ravel() - u_exact) / np.linalg.norm(u_exact)
print(f"Relative L² error at t={T}: {rel_l2:.2e}")
```

## Full runnable script

See {download}`examples/heat_2d.py <../../examples/heat_2d.py>`.
