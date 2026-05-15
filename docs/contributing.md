# Contributing to PyPIELM

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/kstruniawski/pypielm.git
cd pypielm
pip install -e ".[dev]"
```

## Running the Test Suite

```bash
pytest tests/ -q
```

With coverage report:

```bash
pytest tests/ --cov=pypielm --cov-report=term-missing
```

## Code Style

- **Ruff** for linting and import sorting: `ruff check pypielm/`
- **Mypy** for type checking: `mypy pypielm/`
- **Docstrings**: Google style — include `Args:`, `Returns:`, `Raises:`, and
  an `Example::` block for every public function.

Pre-commit hooks are configured in `.pre-commit-config.yaml` (if present).

## Adding a New Model

1. Create your class in the appropriate module (e.g., `pypielm/models/vanilla.py`).
2. Inherit from `BasePIELM` and implement `fit`, `predict`, `score`, and
   `get_feature_matrix`.
3. Decorate the class with `@register("my_model_name")` from
   `pypielm.models.registry`.
4. Add at least one accuracy test in `tests/` solving the 1D Poisson problem
   with relative L² < 1e-2.
5. Export the class in `pypielm/models/__init__.py`.

## Commit Convention

```
feat(step-N): short description
fix: short description
docs: short description
test: short description
refactor: short description
```

## Submitting a Pull Request

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`.
2. Make your changes and ensure all tests pass.
3. Push and open a PR against `main`.
4. The CI pipeline will run lint, test, and docs checks automatically.
