"""Patch extraction (continuous-in-xi) and helpers."""

import jax
import jax.numpy as jnp
from typing import Tuple


def extract_integer_patch(theta: jnp.ndarray, y0: int, x0: int, ph: int, pw: int) -> jnp.ndarray:
    H, W, C = theta.shape
    ys = jnp.clip(jnp.arange(ph) + y0, 0, H - 1)
    xs = jnp.clip(jnp.arange(pw) + x0, 0, W - 1)
    patch = theta[ys[:, None], xs[None, :], :]
    return patch


def O_xi(theta: jnp.ndarray, xi: jnp.ndarray, patch_shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Bilinear continuous patch extractor O_xi via interpolation of four integer-positioned patches.

    Args:
        theta: object array, shape (H, W, C), complex64 preferred
        xi: scan position (2,) as (y, x) float
        patch_shape: (ph, pw)

    Returns:
        patch: (ph, pw, C)
    """
    ph, pw = patch_shape
    y_f = xi[0]
    x_f = xi[1]
    i = jnp.floor(y_f).astype(jnp.int32)
    j = jnp.floor(x_f).astype(jnp.int32)
    dy = y_f - i.astype(jnp.float32)
    dx = x_f - j.astype(jnp.float32)

    p00 = extract_integer_patch(theta, i, j, ph, pw)
    p01 = extract_integer_patch(theta, i, j + 1, ph, pw)
    p10 = extract_integer_patch(theta, i + 1, j, ph, pw)
    p11 = extract_integer_patch(theta, i + 1, j + 1, ph, pw)

    w_a = (1.0 - dy) * (1.0 - dx)
    w_b = (1.0 - dy) * dx
    w_c = dy * (1.0 - dx)
    w_d = dy * dx

    patch = w_a * p00 + w_b * p01 + w_c * p10 + w_d * p11
    return patch


def adjoint_scatter(object_shape: Tuple[int, int, int], xi: jnp.ndarray, backpatch: jnp.ndarray, patch_shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Scatter a backpatch (ph, pw, C) into object-space using bilinear-adjoint weights.

    Args:
        object_shape: (H, W, C)
        xi: (2,) scan position
        backpatch: (ph, pw, C)
        patch_shape: (ph, pw)

    Returns:
        u_grad: (H, W, C) complex array with scattered contributions
    """
    H, W, C = object_shape
    ph, pw = patch_shape

    y_f = xi[0]
    x_f = xi[1]
    i = jnp.floor(y_f).astype(jnp.int32)
    j = jnp.floor(x_f).astype(jnp.int32)
    dy = y_f - i.astype(jnp.float32)
    dx = x_f - j.astype(jnp.float32)

    w_a = (1.0 - dy) * (1.0 - dx)
    w_b = (1.0 - dy) * dx
    w_c = dy * (1.0 - dx)
    w_d = dy * dx

    ys0 = jnp.clip(jnp.arange(ph) + i, 0, H - 1)
    xs0 = jnp.clip(jnp.arange(pw) + j, 0, W - 1)

    ys1 = ys0
    xs1 = jnp.clip(jnp.arange(pw) + (j + 1), 0, W - 1)

    ys2 = jnp.clip(jnp.arange(ph) + (i + 1), 0, H - 1)
    xs2 = xs0

    ys3 = ys2
    xs3 = xs1

    u_grad = jnp.zeros((H, W, C), dtype=backpatch.dtype)

    def scatter(u, ys, xs, weight):
        patch = backpatch * weight
        return u.at[ys[:, None], xs[None, :], :].add(patch)

    u_grad = scatter(u_grad, ys0, xs0, w_a)
    u_grad = scatter(u_grad, ys1, xs1, w_b)
    u_grad = scatter(u_grad, ys2, xs2, w_c)
    u_grad = scatter(u_grad, ys3, xs3, w_d)

    return u_grad
