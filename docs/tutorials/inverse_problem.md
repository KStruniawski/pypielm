# Tutorial: Inverse Problem — Identify PDE Coefficients

An *inverse problem* uses sparse observations of the solution to identify
unknown PDE coefficients. PyPIELM handles this naturally: the observed data
block enters the least-squares system alongside the PDE residual block.

## Problem

Consider the 1D advection-diffusion equation

$$u_t + c \, u_x - \nu \, u_{xx} = 0$$

with *unknown* advection speed $c$ and diffusivity $\nu$.
We have noisy point measurements of $u$ at scattered $(x, t)$ locations and
want to recover $c$ and $\nu$.

## Strategy: Embedded Coefficient Identification

One tractable approach is to treat $c$ and $\nu$ as additional unknowns
appended to the output-weight vector.  This requires augmenting the feature
matrix — an advanced topic.

A simpler baseline uses **BayesianPIELM** with a Gaussian prior over
coefficients and iterative refinement:

```python
from pypielm.models import BayesianPIELM
from pypielm.data.dataset import PIELMDataset

# Suppose `ds` contains X_data (observation locations) and y_data (noisy u values)
model = BayesianPIELM(hidden_dim=200, prior_precision=1.0, seed=42)
model.fit(ds, pde_operator=advection_diffusion_op)

# Posterior over output weights gives uncertainty quantification
beta_mean = model.beta_mean   # (hidden_dim,) tensor
beta_cov  = model.beta_cov    # (hidden_dim, hidden_dim) tensor
```

## Reference

For a full derivation of embedded coefficient identification in ELMs, see:

> Struniawski, K. (2026). *PyPIELM: A Unified and Reproducible Framework
> for Physics-Informed Extreme Learning Machines.*
