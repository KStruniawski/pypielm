"""CLI entry point for PyPIELM.

Usage::

    python -m pypielm run    --config experiment.yaml
    python -m pypielm sweep  --config sweep.yaml [--parallel N]
    python -m pypielm export --model runs/model.pt --format onnx|torchscript
    python -m pypielm list-models

Sub-command details
-------------------
``run``
    Execute a single experiment defined by a YAML config file.  Writes
    ``model.pt`` and ``results.json`` to ``output_dir``.

``sweep``
    Execute multiple experiments from a YAML file that has a top-level ``sweep``
    key listing config overrides.  Runs are dispatched via
    :class:`concurrent.futures.ProcessPoolExecutor` (``--parallel`` controls
    worker count).  A consolidated ``batch_summary.json`` is written to the
    ``output_dir`` of the first config entry (or the current directory).

``export``
    Export a saved checkpoint to ONNX or TorchScript.

``list-models``
    Print all model names available in the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sub-command: run
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    from pypielm.utils.config import load_config, run_experiment

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir

    print(f"Running experiment: model={config.model!r}, device={config.device!r}, seed={config.seed}")
    result = run_experiment(config)

    metrics = result["metrics"]
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Artifacts: {result['artifacts']}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: sweep
# ---------------------------------------------------------------------------

def _run_single_sweep_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Worker function for ProcessPoolExecutor — must be top-level for pickling."""
    import tempfile
    from pathlib import Path as _Path

    import yaml

    from pypielm.utils.config import load_config, run_experiment

    # Write entry to a temp YAML, load it, run
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(entry, tmp)
        tmp_path = _Path(tmp.name)

    try:
        config = load_config(tmp_path)
        result = run_experiment(config)
        return {"status": "ok", **result}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "config": entry}
    finally:
        tmp_path.unlink(missing_ok=True)


def _cmd_sweep(args: argparse.Namespace) -> int:
    import concurrent.futures

    import yaml

    sweep_path = Path(args.config)
    if not sweep_path.exists():
        print(f"Sweep config not found: {sweep_path}", file=sys.stderr)
        return 1

    with sweep_path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    entries: list[dict[str, Any]] = raw.get("sweep", [])
    if not entries:
        print(
            "No 'sweep' key found in config, or it is empty.\n"
            "Expected format:\n\n"
            "  sweep:\n"
            "    - model: vanilla_pielm\n"
            "      model_kwargs: {hidden_dim: 100}\n"
            "    - model: core_pielm\n"
            "      model_kwargs: {hidden_dim: 200}\n",
            file=sys.stderr,
        )
        return 1

    parallel = max(1, args.parallel)
    print(f"Sweeping {len(entries)} configs with {parallel} worker(s)...")

    results: list[dict[str, Any]] = []
    if parallel == 1:
        for i, entry in enumerate(entries, 1):
            print(f"  [{i}/{len(entries)}] {entry.get('model', '?')}")
            results.append(_run_single_sweep_entry(entry))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_single_sweep_entry, e): i
                       for i, e in enumerate(entries, 1)}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"status": "error", "error": str(exc)}
                print(f"  [{idx}/{len(entries)}] done — status={res.get('status')}")
                results.append(res)

    # Determine output directory
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        entries[0].get("output_dir", "runs/sweep/")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "batch_summary.json"
    with summary_path.open("w") as fh:
        json.dump(results, fh, indent=2)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_err = len(results) - n_ok
    print(f"\nSweep complete: {n_ok} OK, {n_err} errors.")
    print(f"Summary written to: {summary_path}")
    return 0 if n_err == 0 else 1


# ---------------------------------------------------------------------------
# Sub-command: export
# ---------------------------------------------------------------------------

def _cmd_export(args: argparse.Namespace) -> int:

    from pypielm.io.checkpoint import load_model
    from pypielm.io.export import to_onnx, to_torchscript

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device=args.device)

    fmt = args.format.lower()
    input_dim = args.input_dim

    if fmt == "onnx":
        out_path = ckpt_path.with_suffix(".onnx")
        try:
            to_onnx(model, out_path, input_dim=input_dim)
            print(f"ONNX model written to: {out_path}")
        except ImportError as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            return 1
    elif fmt in {"torchscript", "ts"}:
        out_path = ckpt_path.with_suffix(".jit.pt")
        to_torchscript(model, out_path, input_dim=input_dim, method=args.ts_method)
        print(f"TorchScript model written to: {out_path}")
    else:
        print(f"Unknown format '{args.format}'. Choose 'onnx' or 'torchscript'.", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Sub-command: list-models
# ---------------------------------------------------------------------------

def _cmd_list_models(_args: argparse.Namespace) -> int:
    import pypielm.models  # noqa: F401 — ensures all models self-register
    from pypielm.models.registry import MODEL_REGISTRY

    print(f"Registered models ({len(MODEL_REGISTRY)}):")
    for name in sorted(MODEL_REGISTRY):
        cls = MODEL_REGISTRY[name]
        print(f"  {name:<35} ({cls.__module__}.{cls.__qualname__})")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pypielm",
        description="PyPIELM: Physics-Informed Extreme Learning Machines CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single experiment from a YAML config.")
    p_run.add_argument("--config", required=True, metavar="FILE",
                       help="Path to the experiment YAML config.")
    p_run.add_argument("--device", default=None, metavar="DEV",
                       help="Override device (e.g. 'cpu', 'cuda', 'mps').")
    p_run.add_argument("--seed", type=int, default=None,
                       help="Override random seed.")
    p_run.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                       help="Override output directory.")

    # ---- sweep --------------------------------------------------------------
    p_sweep = sub.add_parser(
        "sweep",
        help="Run multiple experiments from a YAML sweep config.",
    )
    p_sweep.add_argument("--config", required=True, metavar="FILE",
                         help="Path to the sweep YAML (must have a top-level 'sweep' list).")
    p_sweep.add_argument("--parallel", type=int, default=1, metavar="N",
                         help="Number of parallel worker processes (default: 1).")
    p_sweep.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                         help="Override output directory for batch_summary.json.")

    # ---- export -------------------------------------------------------------
    p_export = sub.add_parser("export", help="Export a saved model to ONNX or TorchScript.")
    p_export.add_argument("--model", required=True, metavar="FILE",
                          help="Path to the model checkpoint (.pt).")
    p_export.add_argument("--format", required=True, metavar="FMT",
                          choices=["onnx", "torchscript", "ts"],
                          help="Export format: 'onnx' or 'torchscript'.")
    p_export.add_argument("--device", default="cpu", metavar="DEV",
                          help="Device to load the model onto (default: cpu).")
    p_export.add_argument("--input-dim", type=int, default=1, dest="input_dim",
                          metavar="D",
                          help="Spatial input dimension d (default: 1).")
    p_export.add_argument("--ts-method", default="trace", dest="ts_method",
                          choices=["trace", "script"],
                          help="TorchScript method (default: trace).")

    # ---- list-models --------------------------------------------------------
    sub.add_parser("list-models", help="List all registered model names.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate sub-command handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "run": _cmd_run,
        "sweep": _cmd_sweep,
        "export": _cmd_export,
        "list-models": _cmd_list_models,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
