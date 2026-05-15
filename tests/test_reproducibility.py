"""Tests for pypielm.utils.reproducibility."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from pypielm.utils.reproducibility import get_device, seed_everything


class TestSeedEverything:
    def test_torch_cpu_reproducible(self):
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        seed_everything(1)
        t1 = torch.randn(10)
        seed_everything(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)

    def test_numpy_reproducible(self):
        seed_everything(7)
        a1 = np.random.rand(10)
        seed_everything(7)
        a2 = np.random.rand(10)
        assert np.allclose(a1, a2)

    def test_python_random_reproducible(self):
        seed_everything(99)
        v1 = [random.random() for _ in range(10)]
        seed_everything(99)
        v2 = [random.random() for _ in range(10)]
        assert v1 == v2

    def test_returns_none(self):
        result = seed_everything(0)
        assert result is None

    def test_does_not_raise(self):
        """seed_everything must not raise for any non-negative integer seed."""
        for s in [0, 1, 12345, 2**31 - 1]:
            seed_everything(s)

    def test_deterministic_false(self):
        """deterministic=False should still seed without raising."""
        seed_everything(42, deterministic=False)
        t1 = torch.randn(5)
        seed_everything(42, deterministic=False)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)

    @pytest.mark.mps
    def test_mps_reproducible(self):
        """Same seed produces identical tensors on MPS."""
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS not available.")
        seed_everything(42)
        t1 = torch.randn(10, device="mps")
        seed_everything(42)
        t2 = torch.randn(10, device="mps")
        assert torch.allclose(t1.cpu(), t2.cpu())


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_prefer_cpu(self):
        device = get_device(prefer_cuda=False, prefer_mps=False)
        assert device.type == "cpu"

    def test_no_accelerator_returns_cpu(self):
        """When neither CUDA nor MPS is available (or both disabled), return CPU."""
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            pytest.skip("An accelerator is present; testing CPU-only path skipped.")
        device = get_device(prefer_cuda=True, prefer_mps=True)
        assert device.type == "cpu"

    @pytest.mark.cuda
    def test_cuda_when_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available.")
        device = get_device(prefer_cuda=True)
        assert device.type == "cuda"

    @pytest.mark.mps
    def test_mps_when_available(self):
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS not available.")
        # CUDA takes priority, so disable it for this test
        device = get_device(prefer_cuda=False, prefer_mps=True)
        assert device.type == "mps"

    @pytest.mark.mps
    def test_auto_prefers_mps_over_cpu(self):
        """When CUDA absent but MPS present, auto-selection picks MPS."""
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS not available.")
        if torch.cuda.is_available():
            pytest.skip("CUDA present; CUDA takes priority over MPS.")
        device = get_device(prefer_cuda=True, prefer_mps=True)
        assert device.type == "mps"

    @pytest.mark.cuda
    def test_cuda_beats_mps(self):
        """When both CUDA and MPS are available, CUDA wins."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available.")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available.")
        device = get_device(prefer_cuda=True, prefer_mps=True)
        assert device.type == "cuda"

    def test_prefer_mps_false_returns_cpu_when_no_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present; test is for no-CUDA machines.")
        device = get_device(prefer_cuda=True, prefer_mps=False)
        assert device.type == "cpu"

