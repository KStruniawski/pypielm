"""Collocation point samplers and domain descriptors.

Public API::

    from pypielm.pde.collocation import (
        BoxDomain, UniformSampler, LHSSampler, AdaptiveSampler, GridSampler
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Domain descriptors
# ---------------------------------------------------------------------------

@dataclass
class BoxDomain:
    """Axis-aligned bounding box in R^d.

    Args:
        lb: Lower bounds, shape ``(d,)``.
        ub: Upper bounds, shape ``(d,)``.

    Example::

        domain = BoxDomain(lb=[0.0, 0.0], ub=[1.0, 1.0])
    """

    lb: list[float] | torch.Tensor
    ub: list[float] | torch.Tensor

    def __post_init__(self) -> None:
        self.lb = torch.as_tensor(self.lb, dtype=torch.float64)
        self.ub = torch.as_tensor(self.ub, dtype=torch.float64)

    @property
    def dim(self) -> int:
        """Spatial dimension d."""
        return int(self.lb.shape[0])


class UnionDomain:
    """Union of multiple :class:`BoxDomain` objects.

    Args:
        domains: List of BoxDomain objects forming the union.
    """

    def __init__(self, domains: list[BoxDomain]) -> None:
        if not domains:
            raise ValueError("UnionDomain requires at least one domain.")
        self.domains = domains

    @property
    def dim(self) -> int:
        return self.domains[0].dim


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class UniformSampler:
    """Sample collocation points uniformly at random within a :class:`BoxDomain`.

    Args:
        domain: The spatial domain.
        n_points: Number of collocation points to sample.
        seed: Random seed.
    """

    def __init__(
        self,
        domain: BoxDomain,
        n_points: int = 1000,
        seed: int = 42,
    ) -> None:
        self.domain = domain
        self.n_points = n_points
        self.seed = seed

    def sample(self) -> torch.Tensor:
        """Draw ``n_points`` uniform samples from ``domain``.

        Returns:
            Tensor of shape ``(n_points, d)``.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        lb = self.domain.lb  # (d,)
        ub = self.domain.ub  # (d,)
        # uniform in [0, 1) then scale
        unit = torch.rand(self.n_points, self.domain.dim, dtype=torch.float64, generator=gen)
        return lb + unit * (ub - lb)


class LHSSampler:
    """Latin Hypercube Sampling (LHS) within a :class:`BoxDomain`.

    LHS ensures better space-filling than pure uniform sampling.  Uses
    ``scipy.stats.qmc.LatinHypercube`` when available; falls back to a
    stratified uniform sampler otherwise.

    Args:
        domain: The spatial domain.
        n_points: Number of collocation points.
        seed: Random seed.
    """

    def __init__(
        self,
        domain: BoxDomain,
        n_points: int = 1000,
        seed: int = 42,
    ) -> None:
        self.domain = domain
        self.n_points = n_points
        self.seed = seed

    def sample(self) -> torch.Tensor:
        """Draw ``n_points`` LHS samples.

        Returns:
            Tensor of shape ``(n_points, d)``.
        """
        lb = self.domain.lb.numpy()
        ub = self.domain.ub.numpy()
        d = self.domain.dim
        n = self.n_points

        try:
            from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
            sampler = LatinHypercube(d=d, seed=self.seed)
            unit_sample = sampler.random(n=n)  # (n, d) in [0, 1)
            scaled = qmc_scale(unit_sample, lb, ub)
        except ImportError:
            # Stratified fallback: split each axis into n strata
            import numpy as np
            rng = np.random.default_rng(self.seed)
            strata = (rng.random((n, d)) + np.arange(n)[:, None]) / n  # stratified
            for j in range(d):
                rng.shuffle(strata[:, j])
            scaled = lb + strata * (ub - lb)

        return torch.tensor(scaled, dtype=torch.float64)


class AdaptiveSampler:
    """Residual-guided adaptive collocation sampler.

    Samples a large candidate set, evaluates the provided ``residual_fn``,
    and returns points concentrated in high-residual regions (top
    ``refine_ratio`` fraction by residual magnitude) padded with uniform
    samples.

    Args:
        domain: The spatial domain.
        residual_fn: Callable ``f(X: Tensor) → Tensor`` returning scalar
            residuals of shape ``(N,)`` or ``(N, 1)`` for input of shape
            ``(N, d)``.
        n_points: Number of collocation points to return.
        refine_ratio: Fraction of returned points that are residual-guided
            (the rest are uniform samples).
        n_candidates: Number of candidates to evaluate before selecting.
        seed: Random seed.
    """

    def __init__(
        self,
        domain: BoxDomain,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        n_points: int = 1000,
        refine_ratio: float = 0.5,
        n_candidates: int = 10_000,
        seed: int = 42,
    ) -> None:
        if not (0.0 < refine_ratio <= 1.0):
            raise ValueError("refine_ratio must be in (0, 1].")
        self.domain = domain
        self.residual_fn = residual_fn
        self.n_points = n_points
        self.refine_ratio = refine_ratio
        self.n_candidates = n_candidates
        self.seed = seed

    def sample(self) -> torch.Tensor:
        """Sample collocation points with residual-guided refinement.

        Returns:
            Tensor of shape ``(n_points, d)``.
        """
        # Draw a large candidate set uniformly
        base_sampler = UniformSampler(self.domain, self.n_candidates, self.seed)
        candidates = base_sampler.sample()  # (n_candidates, d)

        # Evaluate residuals
        with torch.no_grad():
            res = self.residual_fn(candidates)
        res = res.reshape(-1).abs()  # (n_candidates,)

        n_guided = int(self.n_points * self.refine_ratio)
        n_uniform = self.n_points - n_guided

        # Top-k by residual magnitude (guided points)
        _, guided_idx = torch.topk(res, min(n_guided, len(res)))
        guided = candidates[guided_idx]

        if n_uniform > 0:
            unif_sampler = UniformSampler(
                self.domain, n_uniform, self.seed + 1
            )
            uniform_pts = unif_sampler.sample()
            return torch.cat([guided, uniform_pts], dim=0)
        return guided


class GridSampler:
    """Structured Cartesian grid sampler (1D, 2D, or higher).

    Useful for finite-difference baselines and structured visualisation.

    Args:
        domain: The spatial domain.
        nx: Number of grid points along axis 0.
        ny: Number of grid points along axis 1 (ignored for 1D domains).
    """

    def __init__(
        self,
        domain: BoxDomain,
        nx: int = 64,
        ny: int = 64,
    ) -> None:
        self.domain = domain
        self.nx = nx
        self.ny = ny

    def sample(self) -> torch.Tensor:
        """Return all grid points as a tensor.

        Returns:
            Tensor of shape ``(N_total, d)`` where ``N_total = nx`` (1D) or
            ``nx * ny`` (2D).
        """
        d = self.domain.dim
        lb = self.domain.lb
        ub = self.domain.ub

        if d == 1:
            pts = torch.linspace(lb[0].item(), ub[0].item(), self.nx, dtype=torch.float64)
            return pts.unsqueeze(1)  # (nx, 1)
        elif d == 2:
            xs = torch.linspace(lb[0].item(), ub[0].item(), self.nx, dtype=torch.float64)
            ys = torch.linspace(lb[1].item(), ub[1].item(), self.ny, dtype=torch.float64)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        else:
            # Higher dimensions: axis-aligned linspace, all combinations
            linspaces = [
                torch.linspace(lb[i].item(), ub[i].item(), self.nx, dtype=torch.float64)
                for i in range(d)
            ]
            grids = torch.meshgrid(*linspaces, indexing="ij")
            return torch.stack([g.reshape(-1) for g in grids], dim=1)

