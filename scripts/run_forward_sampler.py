"""Run forward (measurement-guided) ULA sampling given a measurement history.

Usage (example):
    PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2 /path/to/python scripts/run_forward_sampler.py \
        --ckpt /path/to/ckpt.pkl --history /path/to/history.pkl --out out.npz

`history` should be a pickle file containing a dict with keys:
  - 'xis': array (J,2)
  - 'f': array (J, ph, pw, C)
  - 'probe': array (ph, pw, C)
  - 'patch_shape': (ph, pw)

If `--ckpt` is given the script will try to construct a prior score from the
diffusion model; otherwise the sampler will run using only measurement guidance.
"""

import argparse
import pickle
import os

import jax
import jax.numpy as jnp
import numpy as np

from ptycho.sampler import generate_ptycho_posterior_samples_ula
from diffusion.model import create_complexUnet, score_apply


def load_ckpt(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    params_ckpt = d.get('ema_params', d.get('params'))
    return params_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--history', type=str, required=True)
    p.add_argument('--out', type=str, default='posterior_samples.npz')
    p.add_argument('--P', type=int, default=8)
    p.add_argument('--n_steps', type=int, default=200)
    p.add_argument('--step_size', type=float, default=1e-4)
    p.add_argument('--measurement_weight', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    with open(args.history, 'rb') as f:
        hist = pickle.load(f)

    xis = jnp.array(hist['xis'])
    f_counts = jnp.array(hist['f'])
    probe = jnp.array(hist['probe'])
    patch_shape = tuple(hist.get('patch_shape', probe.shape[:2]))

    # prepare prior score if checkpoint provided
    prior_params = None
    prior_apply = None
    if args.ckpt is not None:
        params_ckpt = load_ckpt(args.ckpt)
        # Infer reasonable model statics via create_complexUnet
        rng = jax.random.PRNGKey(0)
        _, apply_fn = create_complexUnet(rng, input_shape=(patch_shape[0], patch_shape[1], probe.shape[2], 1))
        # use tuple (params, apply_fn) accepted by score_apply
        prior_params = (params_ckpt, apply_fn)

    # wrapper prior score apply that matches expected signature: (params, x, t)
    def prior_score_wrapper(params, x, t):
        # if no params provided use score_apply stub
        if params is None:
            return score_apply(None, x, t if t is not None else 0.5)
        return score_apply(params, x, t if t is not None else 0.5)

    key = jax.random.PRNGKey(args.seed)

    samples = generate_ptycho_posterior_samples_ula(key, args.P,
                                                   init=None,
                                                   prior_score_apply=(prior_score_wrapper if prior_params is not None else None),
                                                   prior_params=prior_params,
                                                   xis=xis,
                                                   f_measurements=f_counts,
                                                   probe=probe,
                                                   patch_shape=patch_shape,
                                                   n_steps=args.n_steps,
                                                   step_size=args.step_size,
                                                   measurement_weight=args.measurement_weight)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    samples_np = np.array(samples)
    np.savez_compressed(args.out, samples=samples_np)
    print('Saved samples to', args.out)


if __name__ == '__main__':
    main()
