# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24, SciPy ≥ 1.10, PyYAML ≥ 6.0

## From PyPI

```bash
pip install pypielm                   # core only
pip install "pypielm[viz]"            # + matplotlib
pip install "pypielm[viz,bench]"      # + memory profiling
pip install "pypielm[viz,export]"     # + ONNX / onnxruntime
pip install "pypielm[dev]"            # full development environment
```

## From Source

```bash
git clone https://github.com/kstruniawski/pypielm.git
cd pypielm
pip install -e ".[dev]"
```

## GPU Support

PyPIELM inherits GPU support from PyTorch. Install the CUDA-enabled PyTorch
wheel for your platform from [pytorch.org](https://pytorch.org/get-started/locally/)
before installing PyPIELM.

```bash
# Example: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "pypielm[viz]"
```

On Apple Silicon, MPS acceleration is available out of the box with
`device="mps"`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU
for unsupported ops.

## Verify Installation

```python
import pypielm
print(pypielm.__version__)
```
