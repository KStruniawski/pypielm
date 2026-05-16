# PyPIELM

**A Unified and Reproducible Framework for Physics-Informed Extreme Learning Machines**
[![Documentation](https://readthedocs.org/projects/pypielm/badge/?version=latest)](https://pypielm.readthedocs.io/en/latest/)
[![CI](https://github.com/KStruniawski/pypielm/actions/workflows/ci.yml/badge.svg)](https://github.com/KStruniawski/pypielm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/KStruniawski/pypielm/branch/main/graph/badge.svg)](https://codecov.io/gh/KStruniawski/pypielm)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](LICENSE.md)

---

## Overview

PyPIELM is an open-source PyTorch-native library that provides a unified implementation of 26+ Physics-Informed Extreme Learning Machine (PIELM) variants alongside PINN baselines for solving partial differential equations (PDEs). It exposes a scikit-learn-style `fit / predict / score` API so that any PIELM or PINN variant can be used in three lines of code.

### Key Features

| Feature | Detail |
|---------|--------|
| 26+ PIELM variants | VanillaPIELM, BayesianPIELM, GFF-PIELM, DPIELM, LocELM, CurriculumPIELM, NullSpacePIELM, … |
| PINN baselines | VanillaPINN, AdaptivePINN, FourierPINN, MuonPINN |
| Data adapters | CSV, NPZ, PINNacle `.dat`, PDEBench HDF5, `torch.utils.data.Dataset` |
| PDE operators | Autograd Laplacian/gradient, analytic fast paths for tanh/sin, BCs/ICs |
| Reproducibility | YAML experiment configs, CLI, deterministic seeding |
| Export | ONNX, TorchScript |
| Visualisation | 1D/2D solution plots, error maps, Pareto fronts, leaderboard heatmaps |

---

## Installation

```bash
pip install pypielm                  # core only
pip install "pypielm[viz]"           # + matplotlib
pip install "pypielm[viz,bench]"     # + memory profiling
pip install "pypielm[dev]"           # full dev environment
```

From source:

```bash
git clone https://github.com/KStruniawski/pypielm.git
cd pypielm
pip install -e ".[dev]"
```

---

## Quickstart

```python
import pypielm
from pypielm.data import auto_load
from pypielm.models import CorePIELM
from pypielm.pde.operators import AnalyticLaplacian

# Load data (CSV, NPZ, PINNacle .dat, …)
ds = auto_load("data/poisson_classic.dat", source="pinnacle")

# Train model
model = CorePIELM(hidden_dim=300, ridge_lambda=1e-8)
model.fit(ds, pde_operator=AnalyticLaplacian())

# Evaluate
print(model.score(ds.X_test, ds.y_test))  # relative L² error
```

---

## Documentation

Full API reference and tutorials: [pypielm.readthedocs.io](https://pypielm.readthedocs.io)

---

## Citation

If you use PyPIELM in your research, please cite:

```bibtex
@software{struniawski2026pypielm,
  author  = {Struniawski, Karol},
  title   = {{PyPIELM}: A Unified and Reproducible Framework for
             Physics-Informed Extreme Learning Machines},
  year    = {2026},
  url     = {https://github.com/KStruniawski/pypielm},
}
```

---

## Related

- **PyPIELM App** — Streamlit web UI for training, benchmarking and exporting models: [github.com/KStruniawski/pypielm-app](https://github.com/KStruniawski/pypielm-app)

---

## License

CC BY-NC-ND 4.0 © Karol Struniawski
