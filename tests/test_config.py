"""Tests for the YAML config loader, run_experiment, and CLI."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.dump(data))
    return p


# ---------------------------------------------------------------------------
# ExperimentConfig dataclass
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_defaults(self):
        from pypielm.utils.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.model == "core_pielm"
        assert cfg.seed == 42
        assert cfg.device == "cpu"
        assert cfg.output_dir == "runs/"
        assert cfg.model_kwargs == {}
        assert cfg.data == {}
        assert cfg.pde == {}

    def test_custom_values(self):
        from pypielm.utils.config import ExperimentConfig
        cfg = ExperimentConfig(model="vanilla_pielm", seed=7, device="cpu")
        assert cfg.model == "vanilla_pielm"
        assert cfg.seed == 7


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_file_not_found(self, tmp_path):
        from pypielm.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_basic_load(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {
            "model": "vanilla_pielm",
            "model_kwargs": {"hidden_dim": 50},
            "seed": 7,
            "device": "cpu",
            "output_dir": str(tmp_path / "out"),
        }
        cfg_path = _write_yaml(tmp_path, data)
        cfg = load_config(cfg_path)
        assert cfg.model == "vanilla_pielm"
        assert cfg.model_kwargs == {"hidden_dim": 50}
        assert cfg.seed == 7
        assert cfg.device == "cpu"

    def test_defaults_from_empty_yaml(self, tmp_path):
        from pypielm.utils.config import load_config
        # Write a YAML with only model field (rest defaults)
        p = tmp_path / "minimal.yaml"
        p.write_text("model: core_pielm\n")
        cfg = load_config(p)
        assert cfg.model == "core_pielm"
        assert cfg.seed == 42

    def test_invalid_model_raises(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {"model": "nonexistent_model_xyz"}
        cfg_path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="not found in registry"):
            load_config(cfg_path)

    def test_invalid_device_raises(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {"model": "vanilla_pielm", "device": "tpu"}
        cfg_path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="device"):
            load_config(cfg_path)

    def test_nonexistent_data_path_raises(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {
            "model": "vanilla_pielm",
            "data": {"path": "/no/such/file.csv"},
        }
        cfg_path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="Data path does not exist"):
            load_config(cfg_path)

    def test_valid_cuda_device(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {"model": "vanilla_pielm", "device": "cuda:0"}
        cfg_path = _write_yaml(tmp_path, data)
        # Should NOT raise — cuda:N is valid even if CUDA is absent at runtime
        cfg = load_config(cfg_path)
        assert cfg.device == "cuda:0"

    def test_invalid_sampler_raises(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {
            "model": "vanilla_pielm",
            "pde": {"collocation": "BogusS"},
        }
        cfg_path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="sampler"):
            load_config(cfg_path)

    def test_pde_block_parsed(self, tmp_path):
        from pypielm.utils.config import load_config
        data = {
            "model": "vanilla_pielm",
            "pde": {"operator": "laplacian", "collocation": "LHSSampler", "n_collocation": 200},
        }
        cfg_path = _write_yaml(tmp_path, data)
        cfg = load_config(cfg_path)
        assert cfg.pde["operator"] == "laplacian"
        assert cfg.pde["n_collocation"] == 200


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment:
    """Integration tests: actually runs a tiny experiment end-to-end."""

    def test_synthetic_data_run(self, tmp_path):
        """run_experiment with no data.path uses synthetic dataset."""
        from pypielm.utils.config import ExperimentConfig, run_experiment

        out = tmp_path / "out"
        cfg = ExperimentConfig(
            model="vanilla_pielm",
            model_kwargs={"hidden_dim": 50},
            data={"n_samples": 100},
            seed=42,
            device="cpu",
            output_dir=str(out),
        )
        result = run_experiment(cfg)

        assert "metrics" in result
        assert "rel_l2" in result["metrics"]
        assert "fit_time_s" in result["metrics"]
        assert "artifacts" in result
        # results.json must have been written
        results_json = out / "results.json"
        assert results_json.exists()
        saved = json.loads(results_json.read_text())
        assert "metrics" in saved
        assert "rel_l2" in saved["metrics"]

    def test_result_keys(self, tmp_path):
        from pypielm.utils.config import ExperimentConfig, run_experiment

        cfg = ExperimentConfig(
            model="core_pielm",
            model_kwargs={"hidden_dim": 30},
            data={"n_samples": 80},
            seed=1,
            device="cpu",
            output_dir=str(tmp_path / "run_keys"),
        )
        result = run_experiment(cfg)
        assert set(result.keys()) == {"metrics", "config", "artifacts"}
        assert result["config"]["model"] == "core_pielm"
        assert result["config"]["seed"] == 1

    def test_metrics_are_finite(self, tmp_path):
        import math
        from pypielm.utils.config import ExperimentConfig, run_experiment

        cfg = ExperimentConfig(
            model="vanilla_pielm",
            model_kwargs={"hidden_dim": 50},
            data={"n_samples": 100},
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "metrics_check"),
        )
        result = run_experiment(cfg)
        for k, v in result["metrics"].items():
            if isinstance(v, float):
                assert math.isfinite(v), f"metric {k} is not finite: {v}"

    def test_model_checkpoint_saved(self, tmp_path):
        from pypielm.utils.config import ExperimentConfig, run_experiment

        out = tmp_path / "ckpt_test"
        cfg = ExperimentConfig(
            model="vanilla_pielm",
            model_kwargs={"hidden_dim": 30},
            data={"n_samples": 60},
            seed=42,
            device="cpu",
            output_dir=str(out),
        )
        result = run_experiment(cfg)
        assert (out / "model.pt").exists()


# ---------------------------------------------------------------------------
# CLI: python -m pypielm
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        """Run the CLI main() and capture stdout/stderr."""
        import io
        import sys as _sys
        from pypielm.__main__ import main

        old_out, old_err = _sys.stdout, _sys.stderr
        out_buf, err_buf = io.StringIO(), io.StringIO()
        _sys.stdout, _sys.stderr = out_buf, err_buf
        try:
            rc = main(argv)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 0
        finally:
            _sys.stdout, _sys.stderr = old_out, old_err
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_no_args_prints_help(self):
        rc, out, err = self._run([])
        assert rc == 0
        assert "COMMAND" in out or "usage" in out.lower() or "COMMAND" in err

    def test_list_models(self):
        rc, out, _ = self._run(["list-models"])
        assert rc == 0
        assert "vanilla_pielm" in out
        assert "core_pielm" in out

    def test_run_synthetic(self, tmp_path):
        cfg_data = {
            "model": "vanilla_pielm",
            "model_kwargs": {"hidden_dim": 30},
            "data": {"n_samples": 60},
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "cli_run"),
        }
        cfg_path = _write_yaml(tmp_path, cfg_data)
        rc, out, err = self._run(["run", "--config", str(cfg_path)])
        assert rc == 0, f"CLI run failed.\nstdout:\n{out}\nstderr:\n{err}"
        assert "rel_l2" in out or (tmp_path / "cli_run" / "results.json").exists()

    def test_run_missing_config(self, tmp_path):
        rc, out, err = self._run(["run", "--config", str(tmp_path / "missing.yaml")])
        assert rc != 0
        assert "not found" in err.lower() or "error" in err.lower()

    def test_run_override_seed(self, tmp_path):
        cfg_data = {
            "model": "vanilla_pielm",
            "model_kwargs": {"hidden_dim": 30},
            "data": {"n_samples": 60},
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "seed_override"),
        }
        cfg_path = _write_yaml(tmp_path, cfg_data)
        rc, out, err = self._run(["run", "--config", str(cfg_path), "--seed", "99"])
        assert rc == 0, f"CLI run with seed override failed.\nstderr:\n{err}"

    def test_sweep_no_key_fails(self, tmp_path):
        """A sweep YAML without a 'sweep' key should fail gracefully."""
        cfg_data = {"model": "vanilla_pielm"}
        cfg_path = _write_yaml(tmp_path, cfg_data)
        rc, out, err = self._run(["sweep", "--config", str(cfg_path)])
        assert rc != 0

    def test_sweep_single_entry(self, tmp_path):
        sweep_data = {
            "sweep": [
                {
                    "model": "vanilla_pielm",
                    "model_kwargs": {"hidden_dim": 30},
                    "data": {"n_samples": 60},
                    "seed": 42,
                    "device": "cpu",
                    "output_dir": str(tmp_path / "sweep_out"),
                }
            ]
        }
        cfg_path = tmp_path / "sweep.yaml"
        cfg_path.write_text(yaml.dump(sweep_data))
        rc, out, err = self._run([
            "sweep", "--config", str(cfg_path),
            "--output-dir", str(tmp_path / "sweep_summary"),
        ])
        assert rc == 0, f"Sweep failed.\nstdout:\n{out}\nstderr:\n{err}"
        summary = tmp_path / "sweep_summary" / "batch_summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert len(data) == 1
        assert data[0]["status"] == "ok"

    def test_export_missing_checkpoint(self, tmp_path):
        rc, out, err = self._run([
            "export",
            "--model", str(tmp_path / "no_model.pt"),
            "--format", "onnx",
        ])
        assert rc != 0


# ---------------------------------------------------------------------------
# _resolve_pde_operator
# ---------------------------------------------------------------------------

class TestResolvePDEOperator:
    def test_none_when_no_operator(self):
        from pypielm.utils.config import _resolve_pde_operator
        assert _resolve_pde_operator({}) is None

    def test_function_style_returns_none(self):
        from pypielm.utils.config import _resolve_pde_operator
        for op in ["laplacian", "gradient", "divergence", "advection_term"]:
            assert _resolve_pde_operator({"operator": op}) is None

    def test_analytic_laplacian(self):
        from pypielm.utils.config import _resolve_pde_operator
        from pypielm.pde.operators import AnalyticLaplacian
        obj = _resolve_pde_operator({"operator": "analytic_laplacian"})
        assert isinstance(obj, AnalyticLaplacian)

    def test_unknown_operator_raises(self):
        from pypielm.utils.config import _resolve_pde_operator
        with pytest.raises(ValueError, match="Unknown pde.operator"):
            _resolve_pde_operator({"operator": "bogus_op"})


# ---------------------------------------------------------------------------
# _resolve_sampler
# ---------------------------------------------------------------------------

class TestResolveSampler:
    def test_none_when_absent(self):
        from pypielm.utils.config import _resolve_sampler
        assert _resolve_sampler({}) is None

    def test_uniform_sampler(self):
        from pypielm.utils.config import _resolve_sampler
        from pypielm.pde.collocation import UniformSampler
        s = _resolve_sampler({"collocation": "UniformSampler", "n_collocation": 50})
        assert isinstance(s, UniformSampler)

    def test_lhs_sampler(self):
        from pypielm.utils.config import _resolve_sampler
        from pypielm.pde.collocation import LHSSampler
        s = _resolve_sampler({"collocation": "LHSSampler", "n_collocation": 50})
        assert isinstance(s, LHSSampler)

    def test_grid_sampler(self):
        from pypielm.utils.config import _resolve_sampler
        from pypielm.pde.collocation import GridSampler
        s = _resolve_sampler({"collocation": "GridSampler", "nx": 10})
        assert isinstance(s, GridSampler)

    def test_adaptive_sampler_returns_none(self):
        # AdaptiveSampler requires a residual_fn — returns None without one
        from pypielm.utils.config import _resolve_sampler
        s = _resolve_sampler({"collocation": "AdaptiveSampler"})
        assert s is None

    def test_unknown_sampler_raises(self):
        from pypielm.utils.config import _resolve_sampler
        with pytest.raises(ValueError):
            _resolve_sampler({"collocation": "BogusS"})
