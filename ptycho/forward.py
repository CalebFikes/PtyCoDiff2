"""Forward model helpers: forward_field and batch helper."""

import jax
import jax.numpy as jnp
from typing import Tuple


def forward_field(theta: jnp.ndarray, xi: jnp.ndarray, probe: jnp.ndarray, patch_shape: Tuple[int,int]) -> jnp.ndarray:
    """
    Compute detector field a = FFT2(probe * patch) for a single scan xi.

    Args:
        theta: (H, W, C) complex
        xi: (2,) float position
        probe: (ph, pw, C) complex
        patch_shape: (ph, pw)

    Returns:
        a: (ph, pw, C) complex (FFT of probe*patch)
    """
    from .ops import O_xi

    patch = O_xi(theta, xi, patch_shape)  # (ph, pw, C)
    field = probe * patch
    # FFT over the two spatial dims (assume last dim is channel)
    a = jnp.fft.fft2(field, axes=(0, 1))
    return a


def forward_batch(theta: jnp.ndarray, xis: jnp.ndarray, probe: jnp.ndarray, patch_shape: Tuple[int,int]) -> jnp.ndarray:
    """Vectorized forward_field over multiple scan positions.

    Args:
        theta: (H, W, C)
        xis: (J, 2)
        probe: (ph, pw, C)

    Returns:
        a_js: (J, ph, pw, C)
    """
    return jax.vmap(lambda xi: forward_field(theta, xi, probe, patch_shape))(xis)
