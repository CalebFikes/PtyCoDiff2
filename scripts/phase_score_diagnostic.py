"""Phase-score diagnostic test for implicit phase choice (JAX)

Usage (example):
PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2 /local/scratch/cfikes/PtyCoDiff2/conda-env/bin/python \
  scripts/phase_score_diagnostic.py --ckpt runs/test_ckpt.npz --input_shape 28 28 1 --N_batches 8 --batch_size 16 --t 0.0

This script implements the functions described in the spec and prints summary statistics.
"""
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diffusion.model import load_checkpoint_params, score_apply


def phase_rotate(u, phi):
    # u: complex array shape (B, ...)
    # phi: scalar or (B,)
    # return e^{i phi} * u with broadcasting
    phi = jnp.asarray(phi)
    # ensure phi has shape (B,) if u has batch dim
    if u.ndim >= 1:
        B = u.shape[0]
        if phi.ndim == 0:
            phi = jnp.full((B,), phi)
    exp = jnp.exp(1j * phi)
    # reshape exp to broadcast over u's spatial dims
    reshape = (exp.shape[0],) + (1,) * (u.ndim - 1)
    return exp.reshape(reshape) * u


def real_inner(a, b):
    # a,b: complex arrays same shape (B,...)
    # return real part of inner product per batch element
    sum_axes = tuple(range(1, a.ndim))
    return jnp.real(jnp.sum(jnp.conj(a) * b, axis=sum_axes))


def h_of_phi(score_model, params, u, t, phi, normalize=None):
    # phi may be scalar or shape (B,)
    u_rot = phase_rotate(u, phi)
    score = score_model(params, u_rot, t)
    # h_raw = Re(<score, i u_rot>)
    h_raw = real_inner(score, 1j * u_rot)
    if normalize is None:
        return h_raw
    if normalize == 'n_pix':
        n_pix = float(jnp.prod(jnp.array(u_rot.shape[1:])))
        return h_raw / (n_pix + 1e-12)
    if normalize == 'u_norm':
        u_norm2 = jnp.sum(jnp.abs(u_rot) ** 2, axis=tuple(range(1, u_rot.ndim)))
        return h_raw / (u_norm2 + 1e-12)
    return h_raw


def find_phi_star_grid(score_model, params, u, t, K=256, normalize=None):
    phis = jnp.linspace(0.0, 2 * jnp.pi, K, endpoint=False)
    # h_grid: (K, B)
    h_grid = jax.vmap(lambda phi: h_of_phi(score_model, params, u, t, phi, normalize=normalize))(phis)
    idx = jnp.argmin(jnp.abs(h_grid), axis=0)
    phi0 = phis[idx]
    # gather per-batch h value at chosen idx
    B = u.shape[0]
    h0 = h_grid[idx, jnp.arange(B)]
    return phi0, h0


def refine_phi_star_newton(score_model, params, u, t, phi0, n_steps=8, eps=1e-3, normalize=None):
    # phi0 shape (B,)
    phi = phi0
    for _ in range(n_steps):
        h0 = h_of_phi(score_model, params, u, t, phi, normalize=normalize)
        hp = h_of_phi(score_model, params, u, t, phi + eps, normalize=normalize)
        hm = h_of_phi(score_model, params, u, t, phi - eps, normalize=normalize)
        dh = (hp - hm) / (2 * eps)
        safe = jnp.abs(dh) > 1e-6
        step = jnp.where(safe, h0 / dh, 0.0)
        step = jnp.clip(step, -0.2, 0.2)
        phi = phi - step
        phi = jnp.mod(phi, 2 * jnp.pi)
    h_final = h_of_phi(score_model, params, u, t, phi, normalize=normalize)
    dh_final = (h_of_phi(score_model, params, u, t, phi + eps, normalize=normalize) - h_of_phi(score_model, params, u, t, phi - eps, normalize=normalize)) / (2 * eps)
    return phi, h_final, dh_final


def sample_cn(key, shape, dtype=jnp.complex64):
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, shape, dtype=jnp.float32)
    im = jax.random.normal(ki, shape, dtype=jnp.float32)
    return ((re + 1j * im) / jnp.sqrt(2.0)).astype(dtype)


def default_get_test_batch(rng, batch_size, input_shape):
    # input_shape: (H,W,C) or (N dims...) excluding batch
    shp = (batch_size,) + tuple(input_shape)
    return sample_cn(rng, shp)


def run_phase_score_diagnostic(
    score_model,
    params,
    rng,
    N_batches=16,
    batch_size=32,
    input_shape=(28, 28, 1),
    t=0.0,
    K=256,
    newton_steps=8,
    eps=1e-3,
    get_test_batch_fn=None,
):
    if get_test_batch_fn is None:
        get_test_batch_fn = default_get_test_batch

    all_h = []
    all_dh = []
    all_phi = []

    for _ in range(N_batches):
        rng, subkey = jax.random.split(rng)
        u = get_test_batch_fn(subkey, batch_size, input_shape)
        phi0, _ = find_phi_star_grid(score_model, params, u, t, K=K)
        phi_star, h_star, dh_star = refine_phi_star_newton(score_model, params, u, t, phi0, n_steps=newton_steps, eps=eps)
        all_h.append(jnp.abs(h_star))
        all_dh.append(dh_star)
        all_phi.append(phi_star)

    all_h = jnp.concatenate(all_h)
    all_dh = jnp.concatenate(all_dh)
    all_phi = jnp.concatenate(all_phi)

    stats = {
        "mean_abs_h": float(jnp.mean(all_h)),
        "median_abs_h": float(jnp.median(all_h)),
        "mean_dh": float(jnp.mean(all_dh)),
        "median_dh": float(jnp.median(all_dh)),
        "frac_negative_dh": float(jnp.mean(all_dh < 0.0)),
        "frac_strong_negative_dh": float(jnp.mean(all_dh < -1e-3)),
    }
    return stats, all_phi, all_h, all_dh


def equivariance_check(score_model, params, rng, batch_size, input_shape, t=0.0):
    rng, k1, k2 = jax.random.split(rng, 3)
    u = default_get_test_batch(k1, batch_size, input_shape)
    theta = jax.random.uniform(k2, shape=(batch_size,), minval=0.0, maxval=2 * jnp.pi)

    phi0_u, _ = find_phi_star_grid(score_model, params, u, t)
    phi_star_u, _, _ = refine_phi_star_newton(score_model, params, u, t, phi0_u)

    u_rot = phase_rotate(u, theta)
    phi0_rot, _ = find_phi_star_grid(score_model, params, u_rot, t)
    phi_star_rot, _, _ = refine_phi_star_newton(score_model, params, u_rot, t, phi0_rot)

    # Expect phi_star_rot ≈ phi_star_u - theta  (mod 2pi)
    diff = jnp.mod(phi_star_rot - (phi_star_u - theta) + jnp.pi, 2 * jnp.pi) - jnp.pi
    return float(jnp.mean(jnp.abs(diff))), diff


def build_score_model_from_ckpt(ckpt_path, rng, input_shape=(28, 28, 1)):
    # load params (NPZ saved by training) and construct score_model callable
    params_recon, apply_fn = load_checkpoint_params(str(ckpt_path), rng, input_shape=input_shape)
    # score_apply expects (params, apply_fn)
    def score_model(params_pair, u, t):
        # return score shape matching u
        return score_apply(params_pair, u, t)

    # params_pair: (params_recon, apply_fn)
    return (params_recon, apply_fn), score_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--input_shape', type=int, nargs='+', default=[28, 28, 1],
                   help='Spatial input shape H W C')
    p.add_argument('--N_batches', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--t', type=float, default=0.0)
    p.add_argument('--K', type=int, default=256)
    p.add_argument('--newton_steps', type=int, default=8)
    p.add_argument('--eps', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    input_shape = tuple(args.input_shape)

    params_pair, score_model = build_score_model_from_ckpt(args.ckpt, rng, input_shape=input_shape)

    stats, phi, h, dh = run_phase_score_diagnostic(
        score_model,
        params_pair,
        rng,
        N_batches=args.N_batches,
        batch_size=args.batch_size,
        input_shape=input_shape,
        t=args.t,
        K=args.K,
        newton_steps=args.newton_steps,
        eps=args.eps,
    )

    print("Phase-score diagnostic stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # equivariance check (small)
    mean_abs_diff, diff = equivariance_check(score_model, params_pair, rng, min(8, args.batch_size), input_shape, t=args.t)
    print(f"Equivariance mean abs diff (phi_star_rot vs phi_star - theta): {mean_abs_diff:.6e}")

    # save a small NPZ with raw arrays
    out_npz = Path('runs') / 'phase_score_diag.npz'
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, phi=np.array(phi), h=np.array(h), dh=np.array(dh), diff=np.array(diff))
    print('Saved raw diagnostic arrays to', out_npz)


if __name__ == '__main__':
    main()
