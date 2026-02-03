"""Measurement-guided sampling helpers for ptychography.

Provides:
- `posterior_score_measurement`: Wirtinger gradient of Poisson log-likelihood
  computed via JAX autodiff from stacked real/imag representation.
- `generate_ptycho_posterior_samples_ula`: conditional ULA sampler that
  combines a provided prior score function with measurement score guidance.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple

from diffusion.sampler import cosine_alpha_sigma

from .likelihood import poisson_score


def posterior_score_measurement(u_ri: jnp.ndarray,
                                 f_measurements: jnp.ndarray,
                                 scan_positions: jnp.ndarray,
                                 probe: jnp.ndarray,
                                 patch_shape: Tuple[int, int],
                                 eps_safe: float = 1e-8,
                                 R: float = 1.0) -> jnp.ndarray:
    """Analytic Wirtinger gradient (real-stacked) of log p(f | u).

    Implements the closed-form likelihood gradient (sum over scans) using
    the `poisson_score` function for each scan and summing the contributions.

    Args:
        u_ri: stacked real/imag array shape (2, H, W[, C])
        f_measurements: (J, ph, pw, C) integer counts
        scan_positions: (J, 2)
        probe: probe array (ph, pw, C)
        patch_shape: (ph, pw)

    Returns:
        grad_ri: same shape as `u_ri` (real-valued)
    """
    # reconstruct complex object
    real = u_ri[0]
    imag = u_ri[1]
    u_complex = real + 1j * imag

    # vectorize poisson_score over (xi, f) pairs
    # poisson_score signature: (theta, xi, f, probe, patch_shape, eps_safe, R)
    vmapped = jax.vmap(lambda xi, f: poisson_score(u_complex, xi, f, probe, patch_shape, eps_safe=eps_safe, R=R), in_axes=(0, 0))
    grads_per_scan = vmapped(scan_positions, f_measurements)  # (J, H, W, C) complex
    # Aggregate over scans: use the mean so the measurement guidance scales
    # independently of the number of measurements J (prevents blow-up when J large).
    grad_sum = jnp.mean(grads_per_scan, axis=0)

    # return stacked real/imag gradient (real-valued)
    grad_ri = jnp.stack([jnp.real(grad_sum), jnp.imag(grad_sum)], axis=0)
    return grad_ri


def _complex_to_stacked_realimag(u: jnp.ndarray) -> jnp.ndarray:
    """Convert complex array (H,W,C) or (P,H,W,C) -> stacked real/imag (...,2,...)."""
    if jnp.iscomplexobj(u):
        r = jnp.real(u)
        i = jnp.imag(u)
        return jnp.stack([r, i], axis=0)
    else:
        # already real-stacked
        return u


def _stacked_realimag_to_complex(u_ri: jnp.ndarray) -> jnp.ndarray:
    # u_ri shape: (2, H, W, C) or (P, 2, H, W, C)
    real = u_ri[..., 0, :, :, :] if u_ri.ndim == 5 else u_ri[0]
    imag = u_ri[..., 1, :, :, :] if u_ri.ndim == 5 else u_ri[1]
    return real + 1j * imag


def _measurement_score_complex(u_complex: jnp.ndarray,
                               f_measurements: jnp.ndarray,
                               scan_positions: jnp.ndarray,
                               probe: jnp.ndarray,
                               patch_shape: Tuple[int, int],
                               eps_safe: float = 1e-8,
                               R: float = 1.0) -> jnp.ndarray:
    """Compute measurement score (complex) for a single complex object u_complex.

    Returns complex-shaped gradient with same shape as `u_complex`.
    """
    # convert to stacked real/imag
    u_ri = jnp.stack([jnp.real(u_complex), jnp.imag(u_complex)], axis=0)
    grad_ri = posterior_score_measurement(u_ri, f_measurements, scan_positions, probe, patch_shape, eps_safe, R)
    # reconstruct complex gradient
    g_complex = grad_ri[0] + 1j * grad_ri[1]
    return g_complex


def generate_ptycho_posterior_samples_ula(key: jax.Array,
                                          P: int,
                                          init: jnp.ndarray = None,
                                          prior_score_apply: Callable = None,
                                          prior_params: Any = None,
                                          xis: jnp.ndarray = None,
                                          f_measurements: jnp.ndarray = None,
                                          probe: jnp.ndarray = None,
                                          patch_shape: Tuple[int, int] = None,
                                          n_t: int = 128,
                                          measurement_weight: float = 0.1,
                                          measurement_clip: float = None,
                                          R: float = 1.0,
                                          eps_safe: float = 1e-8) -> jnp.ndarray:
    """Run conditional ULA to sample from posterior p(u | f).

    Args:
        key: PRNGKey
        P: number of parallel chains
        init: optional initial particles shape (P, H, W, C) complex. If None draw CN(0,I).
        prior_score_apply: callable(prior_params, x, t) -> complex score (same shape as x). If None, prior_score=0.
        prior_params: params passed to prior_score_apply
        xis, f_measurements, probe, patch_shape: measurement history (required if measurement_weight>0)
        n_steps: number of ULA iterations
        step_size: ULA step size
        measurement_weight: scalar weight for measurement score
        R: exposure scalar for likelihood

    Returns:
        samples: (P, H, W, C) complex final particles
    """

    # shape inference from inputs
    if init is None:
        # draw CN(0,I): real/imag ~ N(0, 1/2)
        # produce shape (P, H, W, C) where (H,W,C) inferred from probe or measurements
        if probe is not None and patch_shape is not None:
            H, W = patch_shape[0], patch_shape[1]
            C = probe.shape[2]
        else:
            raise ValueError("Cannot infer init shape; provide `init` or `probe`/`patch_shape`")
        key, kr = jax.random.split(key)
        re = jax.random.normal(kr, (P, H, W, C), dtype=jnp.float32)
        key, ki = jax.random.split(key)
        im = jax.random.normal(ki, (P, H, W, C), dtype=jnp.float32)
        x0 = ((re + 1j * im) / jnp.sqrt(2.0)).astype(jnp.complex64)
    else:
        x0 = init

    # prepare time grid and vmap-able measurement score for single particle
    N = int(n_t)
    t_grid = jnp.linspace(0.0, 1.0, N + 1)

    fm = f_measurements
    xs = xis
    pf = probe
    ps = patch_shape
    meas_score_single = lambda u: _measurement_score_complex(u, fm, xs, pf, ps, eps_safe=eps_safe, R=R)
    meas_score_batch = jax.vmap(meas_score_single)

    def step(carry, idx):
        x, key = carry
        k = idx  # iterate indices from N down to 1
        t_k = t_grid[k]

        # VP schedule
        alpha_k, sigma_k = cosine_alpha_sigma(t_k)
        sigma_safe = jnp.maximum(sigma_k, 1e-8)

        # diffusion coefficient g(t) = sigma'(t) = (pi/2) * alpha(t)
        g_t = (jnp.pi / 2.0) * alpha_k

        # time step dt = |t_k - t_{k-1}| (backwards)
        dt = jnp.abs(t_grid[k] - t_grid[k - 1])

        # prior score evaluated at t_k
        if prior_score_apply is None:
            s_prior = 0.0
        else:
            s_prior = prior_score_apply(prior_params, x, t_k)

        # measurement score (no clipping during dynamics)
        if measurement_weight == 0.0:
            s_meas = 0.0
        else:
            s_meas = meas_score_batch(x)

        total_score = s_prior + measurement_weight * s_meas

        # gaussian noise: complex circular with E|z|^2 = 2
        key, kr = jax.random.split(key)
        re = jax.random.normal(kr, x.shape, dtype=jnp.float32)
        key, ki = jax.random.split(key)
        im = jax.random.normal(ki, x.shape, dtype=jnp.float32)
        z = (re + 1j * im)

        # Reverse SDE Euler-Maruyama step consistent with training sampler
        x_new = x + (g_t ** 2) * total_score * dt + g_t * jnp.sqrt(dt) * z

        return (x_new, key), None

    # run scan for n_steps
    indices = jnp.arange(N, 0, -1)
    (x_final, _), _ = jax.lax.scan(step, (x0, key), indices)

    # Final optional clipping / NaN-guard applied only after sampling (preserves Langevin dynamics)
    if measurement_clip is not None:
        # Clip only finite entries to preserve dynamics; set non-finite entries to zero
        real = jnp.real(x_final)
        imag = jnp.imag(x_final)
        real_finite = jnp.isfinite(real)
        imag_finite = jnp.isfinite(imag)
        finite_mask = real_finite & imag_finite
        real_clipped = jnp.where(finite_mask, jnp.clip(real, -measurement_clip, measurement_clip), 0.0)
        imag_clipped = jnp.where(finite_mask, jnp.clip(imag, -measurement_clip, measurement_clip), 0.0)
        x_final = real_clipped + 1j * imag_clipped

    return x_final
