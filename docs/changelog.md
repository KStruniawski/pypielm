# Changelog

All notable changes to PyPIELM are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-05-15

### Added

- Repository scaffold and `pyproject.toml` packaging (Hatchling).
- `core/`: `BasePIELM`, `RandomFeatureMap`, `FourierFeatureMap`,
  `AutogradFeatureMap`; `ridge_solve`, `rrqr_solve`, `bayesian_solve`,
  `tikhonov_solve`; `seed_everything`, `get_device`.
- `data/`: `PIELMDataset`, adapters for CSV, NPZ, PINNacle `.dat`,
  PDEBench HDF5, and `torch.utils.data.Dataset`; `Normalizer`, `FeatureExpander`;
  `auto_load` dispatcher.
- `pde/`: autograd `gradient`, `laplacian`, `divergence`,
  `advection_term`; `AnalyticLaplacian`; `UniformSampler`, `LHSSampler`,
  `GridSampler`, `AdaptiveSampler`; `BoxDomain`, `UnionDomain`; `DirichletBC`,
  `NeumannBC`, `InitialCondition`, `PeriodicBC`.
- 26 PIELM variants: `VanillaPIELM`, `CorePIELM`, `GFFPIELM`,
  `BayesianPIELM`, `DPIELM`, `LocELM`, `DDELMCoarse`, `CurriculumPIELM`,
  `NullSpacePIELM`, `EigPIELM`, `LSE_ELM`, `StefanPIELM`, `FPIELM`,
  `SGE_PIELM`, `RINN`, `RaNN`, `XPIELM`, `PIELM_RVDS`, `TSPIELM`,
  `KAPIELM`, `SoftPartitionKAPIELM`, `NormalEquationELM`,
  `ParameterRetentionELM`, `PiecewiseELM`, `DELM`, `PinnacleELM`; model
  registry with `@register` decorator and `get_model`.
- PINN baselines: `VanillaPINN`, `AdaptivePINN`, `FourierPINN`,
  `MuonPINN`.
- `metrics/`: `rmse`, `mae`, `relative_l2`, `max_error`,
  `r2_score`, `MetricsBundle`; `io/checkpoint.py` (`save_model`, `load_model`);
  `io/export.py` (`to_onnx`, `to_torchscript`).
- `visualization/plots.py`: `plot_solution_1d`, `plot_solution_2d`,
  `plot_training_history`, `plot_pareto`, `plot_leaderboard_heatmap`,
  `save_figure`.
- `benchmarks/`: `perf_profile.py`, `sweep_hidden_dim.py`,
  `sweep_solver.py`, `compare_numpy_torch.py`, `stats_analysis.py`,
  `compare_devices.py`; all support `--platform` argument.
- `utils/config.py` (`ExperimentConfig`, `load_config`,
  `run_experiment`); CLI `pypielm/__main__.py` (`run`, `sweep`, `export`,
  `list-models` subcommands); `batch_summary.json` for parallel sweeps.
- Sphinx documentation scaffold (`docs/`), runnable example
  scripts (`examples/`), updated `README.md`.

### Tests

- 432 tests passing, 17 skipped, 0 failed.
- MPS (Apple Silicon): 77% benchmark success rate (failures are expected for
  gradient-based PINN models and unsupported MPS ops).
- CPU: 98% benchmark success rate.
