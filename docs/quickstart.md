# Quickstart

This page shows the most common usage patterns. All examples run in under a
minute on CPU.

## Solve 1D Poisson with CorePIELM

```python
import math
import numpy as np
from pypielm.data.dataset import PIELMDataset
from pypielm.models import CorePIELM
from pypielm.core.solver import WeightedLinearSystem

# --- Problem: -u''(x) = π² sin(πx),  u(0) = u(1) = 0 ---

N = 200  # collocation points
X_c  = np.random.default_rng(42).uniform(0, 1, (N, 1))
X_bc = np.array([[0.0], [1.0]])
y_bc = np.array([[0.0], [0.0]])

ds = PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc)

def poisson_op(fm, X):
    import torch, math
    H_xx = fm.d2(X, 0)
    rhs   = (math.pi**2) * torch.sin(math.pi * X)
    return WeightedLinearSystem(H_xx, -rhs, weight=1.0)

model = CorePIELM(hidden_dim=300, seed=42)
model.fit(ds, pde_operator=poisson_op)

# Evaluate on a dense grid
import torch
X_test = torch.linspace(0, 1, 300).unsqueeze(1).double()
u_pred = model.predict(X_test)
u_exact = torch.sin(math.pi * X_test)
rel_l2 = ((u_pred - u_exact).norm() / u_exact.norm()).item()
print(f"Relative L² error: {rel_l2:.2e}")   # typically < 1e-3
```

## Load Data from File

```python
from pypielm.data import auto_load

# PINNacle .dat file
ds = auto_load("data/poisson_classic.dat", source="pinnacle")

# CSV file with columns x, u
ds = auto_load("data/measurements.csv", feature_cols=["x"], target_col="u")

# NumPy archive
ds = auto_load("data/solution.npz")
```

## Run an Experiment from YAML

Save the following as `experiment.yaml`:

```yaml
model: core_pielm
model_kwargs:
  hidden_dim: 300
  ridge_lambda: 1.0e-8
data:
  n_samples: 200
seed: 42
device: cpu
output_dir: runs/quickstart/
```

Then run:

```bash
python -m pypielm run --config experiment.yaml
```

This writes `runs/quickstart/model.pt` and `runs/quickstart/results.json`.

## List Available Models

```bash
python -m pypielm list-models
```

## Export to ONNX

```bash
python -m pypielm export --model runs/quickstart/model.pt --format onnx
```

## Sweep Multiple Configs in Parallel

```yaml
# sweep.yaml
sweep:
  - model: vanilla_pielm
    model_kwargs: {hidden_dim: 100}
    data: {n_samples: 200}
    seed: 42
    device: cpu
    output_dir: runs/sweep/v100/
  - model: core_pielm
    model_kwargs: {hidden_dim: 300}
    data: {n_samples: 200}
    seed: 42
    device: cpu
    output_dir: runs/sweep/c300/
```

```bash
python -m pypielm sweep --config sweep.yaml --parallel 2
```
