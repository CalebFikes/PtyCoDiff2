"""Poisson likelihood score in object space (Wirtinger gradient) for a single scan xi."""

import jax
import jax.numpy as jnp
from typing import Tuple

from .forward import forward_field, forward_batch
from .ops import adjoint_scatter


def poisson_score(theta: jnp.ndarray,
                  xi: jnp.ndarray,
                  f: jnp.ndarray,
                  probe: jnp.ndarray,
                  patch_shape: Tuple[int,int],
                  eps_safe: float = 1e-10,
                  R: float = 1.0) -> jnp.ndarray:
    """
    Compute Wirtinger gradient ∇_{theta*} log p(f | theta, xi) as an object-shaped complex array.

    Args:
        theta: (H, W, C) complex
        xi: (2,) float
        f: (ph, pw, C) real/float - observed counts per pixel
        probe: (ph, pw, C) complex
        patch_shape: (ph, pw)
        eps_safe: numerical safeguard added to |a|^2
        R: exposure scalar

    Returns:
        grad_conj: (H, W, C) complex (gradient w.r.t. conjugate variable)
    """
    ph, pw = patch_shape
    # forward
    a = forward_field(theta, xi, probe, patch_shape)  # (ph, pw, C) complex
    abs2 = jnp.real(a * jnp.conj(a))
    # include exposure R consistently with particle_loglik
    lam = R * abs2 + eps_safe

    # r = a * (f/lam - 1)
    ratio = (f / lam) - 1.0
    r = a * ratio

    # backprop to object-space: ifft2(r) * conj(probe).
    # Note: jnp.fft.ifft2 uses an inverse scaling of 1/(ph*pw), so multiply
    # by (ph*pw) to obtain the proper adjoint of the un-normalized FFT.
    backpatch = jnp.fft.ifft2(r, axes=(0, 1)) * jnp.conj(probe) * (ph * pw)

    # adjoint scatter using bilinear-adjoint weights
    u_grad = adjoint_scatter(theta.shape, xi, backpatch, patch_shape)

    # Note: this u_grad is ∂L/∂theta* (Wirtinger). Return as-is.
    return u_grad


def particle_loglik(theta: jnp.ndarray,
                     xis: jnp.ndarray,
                     f_measurements: jnp.ndarray,
                     probe: jnp.ndarray,
                     patch_shape: Tuple[int,int],
                     R: float = 1.0,
                     eps_safe: float = 1e-10) -> jnp.ndarray:
    """Compute joint log-likelihood over J scans for a single particle theta."""
    a_js = forward_batch(theta, xis, probe, patch_shape)  # (J, ph, pw, C)
    abs2 = jnp.real(a_js * jnp.conj(a_js))
    lam = R * abs2 + eps_safe
    return jnp.sum(f_measurements * jnp.log(lam) - lam)
