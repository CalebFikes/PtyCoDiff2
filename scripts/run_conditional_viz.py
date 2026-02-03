"""Run conditional generation and visualize results.

Workflow:
 - load ground truth image from `datasets/mnist/train.npz` (first entry by default)
 - generate J=1000 random scan positions (uniform non-overlapping-ish)
 - compute Poisson measurements using ptycho.forward.forward_batch
 - run conditional ULA using the trained score in ckpt (if provided)
 - draw P=8 posterior samples and save NPZ
 - call examples/visualize_testbed_complex.py to create images
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from ptycho.forward import forward_batch
from ptycho.sampler import _measurement_score_complex
from diffusion.model import create_complexUnet, score_apply
from diffusion.sampler import make_sampler


def load_mnist_first(path):
    data = np.load(path)
    # expect 'images' or 'x' or 'arr_0'
    for k in ['images', 'x', 'arr_0', 'train_images', 'images_train']:
        if k in data:
            imgs = data[k]
            break
    else:
        # fallback: take first array stored
        keys = list(data.keys())
        imgs = data[keys[0]]
    # normalize to [-1,1]
    img = imgs[0].astype(np.float32)
    if img.max() > 2.0:
        img = img / 255.0
    img = img * 2.0 - 1.0
    # ensure shape (H,W,1)
    if img.ndim == 2:
        img = img[..., None]
    return img


def make_random_scan_positions(H, W, ph, pw, J, seed=0):
    rng = np.random.RandomState(seed)
    ys = rng.randint(0, H - ph + 1, size=J)
    xs = rng.randint(0, W - pw + 1, size=J)
    return jnp.stack([ys, xs], axis=1).astype(jnp.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='/local/scratch/cfikes/PtyCoDiff2/weights/attn7/ckpt_100.pkl')
    p.add_argument('--mnist', type=str, default='/local/scratch/cfikes/PtyCoDiff2/datasets/mnist/train.npz')
    p.add_argument('--outdir', type=str, default='runs/conditional_viz')
    p.add_argument('--J', type=int, default=1000)
    p.add_argument('--P', type=int, default=8)
    p.add_argument('--n_steps', type=int, default=128)
    p.add_argument('--step_size', type=float, default=1e-6)
    p.add_argument('--measurement_weight', type=float, default=1.0)
    p.add_argument('--R', type=float, default=1.0)
    p.add_argument('--measurement_clip', type=float, default=1.0,
                   help='Optional final clipping bound for samples (applied after sampling)')
    p.add_argument('--eps_safe', type=float, default=1e-3,
                   help='Numerical floor for forward intensity lambda to stabilize measurement score')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load image
    img = load_mnist_first(args.mnist)
    H, W, C = img.shape

    # use a smaller probe/patch size (typical ptychography) instead of full-field
    # default to 8x8 or smaller if image is smaller
    ph = min(8, H)
    pw = min(8, W)
    patch_shape = (ph, pw)

    # probe (simple uniform probe)
    probe = jnp.ones((ph, pw, C), dtype=jnp.complex64)

    # scan positions
    xis = make_random_scan_positions(H, W, ph, pw, args.J, seed=args.seed)

    # forward to get intensities
    theta_true = jnp.array(img.astype(jnp.complex64))
    a_js = forward_batch(theta_true, xis, probe, patch_shape)  # (J, ph, pw, C)
    lam = jnp.real(a_js * jnp.conj(a_js))

    R = float(args.R)
    rng = np.random.RandomState(args.seed + 1)
    f = rng.poisson(R * np.array(lam)).astype(np.float32)

    # prepare prior score (from ckpt)
    prior_params = None
    if args.ckpt and Path(args.ckpt).exists():
        with open(args.ckpt, 'rb') as fh:
            d = pickle.load(fh)
        params_ckpt = d.get('ema_params', d.get('params'))
        if params_ckpt is None:
            raise SystemExit('No params found in checkpoint')
        # Recreate an apply_fn with statics that match the checkpoint when possible.
        rng0 = jax.random.PRNGKey(0)
        # sensible defaults
        base_ch = 32
        mixing = 0.1
        att_scale = 0.0
        if isinstance(params_ckpt, dict):
            fr = params_ckpt.get('final_real')
            if isinstance(fr, dict) and hasattr(fr.get('w'), 'shape'):
                try:
                    base_ch = int(fr['w'].shape[2])
                except Exception:
                    pass
            try:
                att_scale = float(params_ckpt.get('att_scale', att_scale))
            except Exception:
                pass
            try:
                mixing = float(params_ckpt.get('m1', {}).get('mix', mixing))
            except Exception:
                pass
        _, apply_fn = create_complexUnet(rng0, input_shape=(H, W, C, 1), base_ch=base_ch, mixing=mixing, att_scale=att_scale)
        prior_params = (params_ckpt, apply_fn)

    # Build sampler using the same structure as the unconditional sampler
    cfg = {'n_t': args.n_steps,
           'object_shape': (H, W, C),
           'patch_shape': patch_shape,
           'predicts_eps': True,
           'eps_safe': args.eps_safe}

    # drift function: batch -> batch (P, H, W, C)
    fm = jnp.array(f)
    xis_j = xis
    probe_j = probe
    ps = patch_shape
    R_j = float(R)
    eps_safe_j = float(args.eps_safe)
    meas_w = float(args.measurement_weight)

    from diffusion.sampler import cosine_alpha_sigma

    def drift_fn(x_batch, t):
        if meas_w == 0.0:
            return 0.0
        # compute g(t) consistent with VP sampler: g(t) = (pi/2)*alpha(t)
        alpha_t, sigma_t = cosine_alpha_sigma(t)
        g_t = (jnp.pi / 2.0) * alpha_t
        g2 = (g_t ** 2)
        # vmap over batch axis to compute measurement score per sample
        vm = jax.vmap(lambda u: _measurement_score_complex(u, fm, xis_j, probe_j, ps, eps_safe=eps_safe_j, R=R_j))
        # scale measurement score to match how model score is applied in the sampler
        return meas_w * g2 * vm(x_batch)

    # Choose the model apply function so the sampler receives eps_pred (not already-converted score)
    if args.ckpt and Path(args.ckpt).exists():
        # prior_params was set to (params_ckpt, apply_fn)
        params_ckpt = prior_params[0]
        apply_fn = prior_params[1]
        model_score_apply = lambda p, x, t: apply_fn(p, x, t)
        sampler_params = params_ckpt
    else:
        model_score_apply = score_apply
        sampler_params = None

    sampler = make_sampler(cfg, None, model_score_apply, drift_fn=drift_fn)
    sampler_fn = getattr(sampler, '__wrapped__', sampler)

    # run conditional sampler
    key = jax.random.PRNGKey(args.seed + 2)
    if meas_w == 0.0:
        # call sampler the same way as the unconditional script to avoid tracing/specialization differences
        samples = sampler_fn(sampler_params, key, P=args.P)
    else:
        samples = sampler_fn(sampler_params, key, xi=xis, f=jnp.array(f), probe=probe, P=args.P)

    # Save NPZ: particles (P,H,W,C) complex, measurements, positions
    out_npz = outdir / 'posterior_samples.npz'
    np.savez_compressed(out_npz,
                        particles=np.array(samples),
                        measurements=np.array(f),
                        positions=np.array(xis),
                        weights=np.ones(args.P))

    print('Saved posterior NPZ to', out_npz)

    # Create a sampler-style montage identical to `scripts/run_sampler_ckpt.py`
    try:
        import matplotlib.pyplot as plt

        imgs = np.array(jnp.real(samples))
        imgs = (imgs + 1.0) / 2.0
        imgs = np.clip(imgs, 0.0, 1.0)

        P = imgs.shape[0]
        ncol = 4
        nrow = int(np.ceil(P / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten()
        for i in range(nrow * ncol):
            ax = axes[i]
            ax.axis('off')
            if i < P:
                im = imgs[i, ..., 0]
                ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.tight_layout()
        out_png = outdir / 'sampler_style_montage.png'
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print('Saved sampler-style montage to', out_png)
    except Exception as e:
        print('Failed to produce sampler-style montage:', e)


if __name__ == '__main__':
    main()
