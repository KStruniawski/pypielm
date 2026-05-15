# Tutorial: Reproducible Experiments with YAML and the CLI

PyPIELM's CLI and YAML config system let you reproduce any experiment with a
single command, no Python scripting required.

## YAML Config Format

```yaml
# experiment.yaml
model: core_pielm          # registry name (case-insensitive)
model_kwargs:
  hidden_dim: 300
  ridge_lambda: 1.0e-8
data:
  n_samples: 200           # synthetic sinusoidal dataset (no path needed)
  # path: data/poisson.dat # or load from file
  # source: pinnacle       # adapter hint
pde:
  operator: laplacian      # analytic_laplacian | laplacian | gradient | …
  collocation: LHSSampler  # UniformSampler | LHSSampler | GridSampler
  n_collocation: 500
  domain_lb: [0.0]
  domain_ub: [1.0]
seed: 42
device: cpu
output_dir: runs/example/
```

## Run a Single Experiment

```bash
python -m pypielm run --config experiment.yaml
```

Artifacts written to `runs/example/`:

- `model.pt` — PyTorch checkpoint (weights + config)
- `results.json` — metrics and config snapshot

Override config values from the command line:

```bash
python -m pypielm run --config experiment.yaml --device mps --seed 99
```

## Batch Sweep

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
  - model: bayesian_pielm
    model_kwargs: {hidden_dim: 200}
    data: {n_samples: 200}
    seed: 42
    device: cpu
    output_dir: runs/sweep/b200/
```

```bash
python -m pypielm sweep --config sweep.yaml --parallel 3
```

A `batch_summary.json` is written to `--output-dir` (default: first entry's
`output_dir`) listing status, metrics, and artifact paths for all runs.

## Export a Trained Model

```bash
# ONNX
python -m pypielm export --model runs/example/model.pt --format onnx

# TorchScript (trace)
python -m pypielm export --model runs/example/model.pt --format torchscript

# TorchScript (script)
python -m pypielm export --model runs/example/model.pt \
    --format torchscript --ts-method script
```

## List All Models

```bash
python -m pypielm list-models
```

Output:

```
Registered models (30):
  adaptive_pinn          (pypielm.models.pinn.AdaptivePINN)
  bayesian_pielm         (pypielm.models.bayesian.BayesianPIELM)
  core_pielm             (pypielm.models.vanilla.CorePIELM)
  ...
```
