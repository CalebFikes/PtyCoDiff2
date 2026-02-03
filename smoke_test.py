"""Smoke test to verify sampler runs end-to-end on random data.

Run: python3 smoke_test.py
"""
import jax
import jax.numpy as jnp
from diffusion.model import score_apply
from diffusion.sampler import make_sampler
from ptycho.forward import forward_field
from ptycho.likelihood import particle_loglik, poisson_score


def run_smoke():
    # small object and patch for quick smoke test
    H, W, C = 32, 32, 1
    ph, pw = 8, 8
    object_shape = (H, W, C)
    patch_shape = (ph, pw)

    # random ground truth object and probe
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    theta_true = (jax.random.normal(k1, object_shape) + 1j * jax.random.normal(k2, object_shape)).astype(jnp.complex64)
    probe = jnp.ones((ph, pw, C), dtype=jnp.complex64)

    # choose a continuous xi
    xi = jnp.array([10.3, 12.7], dtype=jnp.float32)

    # generate forward field and sample Poisson counts
    a = forward_field(theta_true, xi, probe, patch_shape)
    lam = jnp.real(a * jnp.conj(a))
    R = 4.0
    rng = jax.random.PRNGKey(1)
    y = jax.random.poisson(rng, R * lam).astype(jnp.float32)

    # sampler config
    N = 16
    t_grid = jnp.linspace(0.0, 1.0, N + 1)
    sigma_grid = jnp.sin(0.5 * jnp.pi * t_grid)
    gamma_grid = (1e-4) * (sigma_grid[1:] / sigma_grid[-1]) ** 2
    cfg = {
        "N": N,
        "t_grid": t_grid,
        "sigma_grid": sigma_grid,
        "gamma_grid": gamma_grid,
        "patch_shape": patch_shape,
        "object_shape": object_shape,
        "tau": 1.0,
        "eps_safe": 1e-8,
        "R": R,
    }

    sampler = make_sampler(cfg, None, score_apply)

    params = None
    key = jax.random.PRNGKey(42)
    # sampler expects a static particle count from cfg; set cfg.P above
    samples = sampler(params, key, xi, y, probe)
    print("Samples shape:", samples.shape)


if __name__ == "__main__":
    run_smoke()
