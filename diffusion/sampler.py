"""Sampler factory that depends on a score_apply from the diffusion module.

This module implements the sampler factory which uses `ptycho` forward and likelihood ops
for guidance but keeps the score/unet dependency in `diffusion`.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional

from ptycho.likelihood import poisson_score


def sample_cn(key: jnp.ndarray, shape, dtype=jnp.complex64):
    """Sample complex Gaussian CN(0,I): real/imag ~ N(0, 1/2)."""
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, shape, dtype=jnp.float32)
    im = jax.random.normal(ki, shape, dtype=jnp.float32)
    return ((re + 1j * im) / jnp.sqrt(2.0)).astype(dtype)


def cosine_alpha_sigma(t: jnp.ndarray):
    """Cosine VP schedule: returns (alpha, sigma)."""
    a = jnp.cos(jnp.pi * 0.5 * t)
    s = jnp.sin(jnp.pi * 0.5 * t)
    return a, s


def make_sampler(cfg: dict, forward_ops: Any, score_apply: Callable, drift_fn: Optional[Callable] = None):
    """
    Return a compiled VP-reverse sampler function sampler(params, key, xi, f, probe).

    This implements the exact VP reverse SDE Euler-Maruyama discretization used to
    invert the training forward process in `train_mnist_stripped.py`.

    Args:
        cfg: dict with keys: N, t_grid (len N+1) or 'n_t' to build uniform grid, object_shape, patch_shape, predicts_eps (bool)
        forward_ops: ptycho forward ops (optional, only needed if guidance/drift uses them)
        score_apply: callable(params, x, t) -> model output. If `cfg.get('predicts_eps', True)` is True
                     the model output is treated as eps_pred and converted to score via -eps/σ(t).
        drift_fn: optional callable(theta, t) -> complex drift array to add (treated as zero if None).
    """
    N = int(cfg.get("N", cfg.get("n_t", 128)))
    if "t_grid" in cfg:
        t_grid = jnp.asarray(cfg["t_grid"])  # length N+1
    else:
        t_grid = jnp.linspace(0.0, 1.0, N + 1)

    patch_shape = tuple(cfg.get("patch_shape", (0, 0)))
    object_shape = tuple(cfg["object_shape"]) if "object_shape" in cfg else (28, 28, 1)
    predicts_eps = bool(cfg.get("predicts_eps", True))
    eps_safe = float(cfg.get("eps_safe", 1e-10))

    H, W, C = object_shape

    # Score-based SDE sampler (no ancestral / DDPM shortcut)

    @jax.jit
    def sampler(params: Any, key: jnp.ndarray, xi: jnp.ndarray = None, f: jnp.ndarray = None, probe: jnp.ndarray = None, P: int = 1):
        """Run VP reverse sampler for P parallel chains. Returns samples shape (P, H, W, C)."""
        # initial sample x_N ~ CN(0, I)
        shape = (P, H, W, C)
        key, subkey = jax.random.split(key)
        x = sample_cn(subkey, shape)

        def step(carry, idx):
            x, key = carry
            k = idx  # idx from N down to 1
            t_k = t_grid[k]

            # VP schedule
            alpha_k, sigma_k = cosine_alpha_sigma(t_k)
            sigma_safe = jnp.maximum(sigma_k, 1e-8)

            # diffusion coefficient g(t) = sigma'(t) = (pi/2) * alpha(t)
            g_t = (jnp.pi / 2.0) * alpha_k

            # time step dt = |t_k - t_{k-1}| (backwards)
            dt = jnp.abs(t_grid[k] - t_grid[k - 1])

            # model output (eps_pred or score)
            model_out = score_apply(params, x, t_k)

            # Score version: convert model output to score if it predicts eps.
            if predicts_eps:
                score = -model_out / sigma_safe
            else:
                score = model_out

            # optional drift/guidance term
            if drift_fn is None:
                drift = 0.0
            else:
                drift = drift_fn(x, t_k)

            # noise term
            key, subkey = jax.random.split(key)
            z = sample_cn(subkey, x.shape)

            # VP reverse SDE Euler-Maruyama step:
            # x_{k-1} = x_k + g(t)^2 * score * dt + drift * dt + g(t) * sqrt(dt) * z
            x_new = x + (g_t ** 2) * score * dt + drift * dt + g_t * jnp.sqrt(dt) * z

            return (x_new, key), None

        indices = jnp.arange(N, 0, -1)
        (x_final, _), _ = jax.lax.scan(step, (x, key), indices)
        return x_final

    return sampler
