# Changelog

All notable changes to PyPIELM are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-15

### Added
- 26+ PIELM variants with PyTorch-native GPU-aware implementation.
- PINN baselines: VanillaPINN, AdaptivePINN, FourierPINN, MuonPINN.
- Data adapters: CSV, NPZ, PINNacle .dat, PDEBench HDF5.
- PDE operators, collocation samplers, BC/IC constraint blocks.
- Metrics: rmse, mae, relative_l2, max_error, r2_score, MetricsBundle.
- YAML experiment config, CLI (run / sweep / export / list-models).
- ONNX and TorchScript export.
- Sphinx documentation, runnable examples, benchmark scripts.
