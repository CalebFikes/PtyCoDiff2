"""Compatibility shim: re-export sampler factory from `diffusion`.

The sampler depends on a score function (the learned prior). The canonical
implementation lives in `diffusion.sampler`. Keep this shim so existing code
importing `ptycho.make_sampler` continues to work.
"""
from diffusion.sampler import make_sampler, cosine_alpha_sigma

"""Merged sampling helpers

This file now contains the original shim (re-exporting `make_sampler`) and the
measurement-guided sampling helpers that were previously implemented in
`ptycho/sampling.py`. The aim is to provide a single import target
`ptycho.sampler` for both the sampler factory and measurement-guided helpers.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple

from .likelihood import poisson_score


def posterior_score_measurement(u_ri: jnp.ndarray,
								 f_measurements: jnp.ndarray,
								 scan_positions: jnp.ndarray,
								 probe: jnp.ndarray,
								 patch_shape: Tuple[int, int],
								 eps_safe: float = 1e-8,
								 R: float = 1.0) -> jnp.ndarray:
	"""Analytic Wirtinger gradient (real-stacked) of log p(f | u).

	Returns stacked real/imag gradient matching the input `u_ri` shape.
	"""
	real = u_ri[0]
	imag = u_ri[1]
	u_complex = real + 1j * imag

	vmapped = jax.vmap(lambda xi, f: poisson_score(u_complex, xi, f, probe, patch_shape, eps_safe=eps_safe, R=R), in_axes=(0, 0))
	grads_per_scan = vmapped(scan_positions, f_measurements)  # (J, H, W, C) complex
	grad_sum = jnp.mean(grads_per_scan, axis=0)
	grad_ri = jnp.stack([jnp.real(grad_sum), jnp.imag(grad_sum)], axis=0)
	return grad_ri


def _complex_to_stacked_realimag(u: jnp.ndarray) -> jnp.ndarray:
	if jnp.iscomplexobj(u):
		r = jnp.real(u)
		i = jnp.imag(u)
		return jnp.stack([r, i], axis=0)
	else:
		return u


def _stacked_realimag_to_complex(u_ri: jnp.ndarray) -> jnp.ndarray:
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
	"""Compute measurement score (complex) for a single complex object.

	Returns a complex-valued gradient with same shape as `u_complex`.
	"""
	u_ri = jnp.stack([jnp.real(u_complex), jnp.imag(u_complex)], axis=0)
	grad_ri = posterior_score_measurement(u_ri, f_measurements, scan_positions, probe, patch_shape, eps_safe, R)
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

	See docstrings in the previous `ptycho/sampling.py` for details.
	"""
	if init is None:
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
		k = idx
		t_k = t_grid[k]

		alpha_k, sigma_k = cosine_alpha_sigma(t_k)
		sigma_safe = jnp.maximum(sigma_k, 1e-8)

		g_t = (jnp.pi / 2.0) * alpha_k
		dt = jnp.abs(t_grid[k] - t_grid[k - 1])

		if prior_score_apply is None:
			s_prior = 0.0
		else:
			s_prior = prior_score_apply(prior_params, x, t_k)

		if measurement_weight == 0.0:
			s_meas = 0.0
		else:
			s_meas = meas_score_batch(x)

		total_score = s_prior + measurement_weight * s_meas

		key, kr = jax.random.split(key)
		re = jax.random.normal(kr, x.shape, dtype=jnp.float32)
		key, ki = jax.random.split(key)
		im = jax.random.normal(ki, x.shape, dtype=jnp.float32)
		z = (re + 1j * im)

		x_new = x + (g_t ** 2) * total_score * dt + g_t * jnp.sqrt(dt) * z

		return (x_new, key), None

	indices = jnp.arange(N, 0, -1)
	(x_final, _), _ = jax.lax.scan(step, (x0, key), indices)

	if measurement_clip is not None:
		real = jnp.real(x_final)
		imag = jnp.imag(x_final)
		real_finite = jnp.isfinite(real)
		imag_finite = jnp.isfinite(imag)
		finite_mask = real_finite & imag_finite
		real_clipped = jnp.where(finite_mask, jnp.clip(real, -measurement_clip, measurement_clip), 0.0)
		imag_clipped = jnp.where(finite_mask, jnp.clip(imag, -measurement_clip, measurement_clip), 0.0)
		x_final = real_clipped + 1j * imag_clipped

	return x_final


__all__ = ["make_sampler", "posterior_score_measurement", "_measurement_score_complex", "generate_ptycho_posterior_samples_ula"]
