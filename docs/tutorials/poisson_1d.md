# Tutorial: Solving 1D Poisson with CorePIELM

This tutorial walks through solving the 1D Poisson equation

$$-u''(x) = \pi^2 \sin(\pi x), \quad x \in [0, 1], \quad u(0) = u(1) = 0$$

whose exact solution is $u(x) = \sin(\pi x)$.

## Step 1 — Set up the dataset

We need interior collocation points and boundary points:

```python
import math
import numpy as np
from pypielm.data.dataset import PIELMDataset

rng = np.random.default_rng(42)
X_c  = rng.uniform(0.0, 1.0, (200, 1))   # interior
X_bc = np.array([[0.0], [1.0]])            # Dirichlet BC
y_bc = np.array([[0.0], [0.0]])

ds = PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc)
```

## Step 2 — Define the PDE operator

The operator encodes the residual $-u'' = \pi^2 \sin(\pi x)$ as a
weighted linear block:

```python
import torch
from pypielm.core.solver import WeightedLinearSystem

def poisson_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    H_xx = fm.d2(X, 0)                                # second derivative of H
    rhs   = (math.pi**2) * torch.sin(math.pi * X)     # right-hand side
    return WeightedLinearSystem(H_xx, -rhs, weight=1.0)
```

## Step 3 — Train the model

```python
from pypielm.models import CorePIELM

model = CorePIELM(hidden_dim=300, ridge_lambda=1e-8, seed=42)
model.fit(ds, pde_operator=poisson_op)
```

## Step 4 — Evaluate

```python
X_test  = torch.linspace(0, 1, 500).unsqueeze(1).double()
u_pred  = model.predict(X_test)
u_exact = torch.sin(math.pi * X_test)

rel_l2 = ((u_pred - u_exact).norm() / u_exact.norm()).item()
print(f"Relative L² error: {rel_l2:.2e}")   # typically < 1e-3
```

## Step 5 — Visualise

```python
from pypielm.visualization import plot_solution_1d, save_figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_solution_1d(
    X_test.numpy(), u_pred.numpy(), u_exact.numpy(),
    title="1D Poisson — CorePIELM",
    ax=ax,
)
save_figure(fig, "poisson_1d_solution.pdf")
```

## Full runnable script

See {download}`examples/poisson_1d.py <../../examples/poisson_1d.py>`.
