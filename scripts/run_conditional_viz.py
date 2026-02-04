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
from diffusion.model import create_complexUnet, score_apply, load_checkpoint_params
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
    # Map to [-1,1] (training uses [-1,1] scale)
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
    p.add_argument('--diagnostics', action='store_true',
                   help='Run short model/score diagnostics before sampling')
    p.add_argument('--force_zero_att_mix', action='store_true',
                   help='Force att_scale and per-level mix values to zero after loading ckpt')
    p.add_argument('--trace_layers', action='store_true',
                   help='Run a layer-by-layer trace of UNet internals and print stats')
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
        rng0 = jax.random.PRNGKey(0)
        # Use loader that handles both legacy (real/imag) and new-format checkpoints
        params_conv, apply_fn = load_checkpoint_params(args.ckpt, rng0, input_shape=(H, W, C))
        prior_params = (params_conv, apply_fn)
        # Optionally force-disable attention/mixing to match training flags
        if args.force_zero_att_mix:
            try:
                # att_scale is a scalar
                if 'att_scale' in params_conv:
                    params_conv['att_scale'] = jnp.array(0.0, dtype=jnp.float32)
                # per-level mixing keys: also zero-out 1x1 mix kernels to avoid WS amplification
                for mk in ['mix1', 'mix2', 'mix3', 'mixb']:
                    if mk in params_conv and isinstance(params_conv[mk], dict) and 'w' in params_conv[mk]:
                        try:
                            params_conv[mk]['w'] = jnp.zeros_like(params_conv[mk]['w'])
                            if 'b' in params_conv[mk]:
                                params_conv[mk]['b'] = jnp.zeros_like(params_conv[mk]['b'])
                        except Exception:
                            pass
                print('Forced att_scale and per-level mix to zero in loaded params')
            except Exception as e:
                print('Failed to force zero att/mix in params:', e)

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
    # Diagnostic checks: run a short per-timestep model/score probe when diagnostics enabled
    if args.diagnostics:
        from diffusion.sampler import sample_cn, cosine_alpha_sigma
        P_dbg = min(4, args.P)
        key, subk = jax.random.split(key)
        x_dbg = sample_cn(subk, (P_dbg, H, W, C))
        # build t grid consistent with sampler config
        t_grid = jnp.linspace(0.0, 1.0, args.n_steps + 1)
        print('Sampler diagnostics: probing model outputs on random CN inputs')
        for k in range(min(6, args.n_steps)):
            t_k = t_grid[k + 1]
            a_k, s_k = cosine_alpha_sigma(t_k)
            # model expects eps_pred (not converted to score) — model_score_apply returns eps_pred
            try:
                me = model_score_apply(sampler_params, x_dbg, t_k)
            except Exception as e:
                print(f'  model_apply error at t_idx={k+1} t={float(t_k):.4f}:', e)
                break
            # convert to score if predicts eps
            sigma_safe = float(jnp.maximum(s_k, 1e-8))
            score_dbg = (-me / sigma_safe) if cfg.get('predicts_eps', True) else me
            def stats(arr):
                import numpy as _np
                r = _np.real(_np.array(arr))
                i = _np.imag(_np.array(arr))
                return (float(r.min()), float(r.max()), float(r.mean()), float(r.std()), float(i.min()), float(i.max()), float(i.mean()), float(i.std()))
            me_stats = stats(me)
            sc_stats = stats(score_dbg)
            print(f' t[{k+1}]={float(t_k):.4f} alpha={float(a_k):.6f} sigma={float(s_k):.6e} sigma_safe={sigma_safe:.6e}')
            print(f'   model eps_pred real(min,max,mean,std)={me_stats[0]:.6e},{me_stats[1]:.6e},{me_stats[2]:.6e},{me_stats[3]:.6e} imag(min,max,mean,std)={me_stats[4]:.6e},{me_stats[5]:.6e},{me_stats[6]:.6e},{me_stats[7]:.6e}')
            print(f'   score real(min,max,mean,std)={sc_stats[0]:.6e},{sc_stats[1]:.6e},{sc_stats[2]:.6e},{sc_stats[3]:.6e}')
        print('End sampler diagnostics')
    # Optional layer trace: replay UNet internals using functions in complex_unet
    if args.trace_layers:
        from diffusion import complex_unet as cunet
        print('Running layer-by-layer trace on zero input')
        P_dbg = min(2, args.P)
        x0 = jnp.zeros((P_dbg, H, W, C), dtype=jnp.complex64)
        t0 = float(0.01)
        # replicate steps from cunet.apply_fn
        x = x0
        had_batch = True
        if x.ndim == 3:
            x = x[None, ...]
            had_batch = False
        x2 = jnp.concatenate([jnp.real(x), jnp.imag(x)], axis=-1)
        t_arr = jnp.full((x.shape[0],), t0, dtype=jnp.float32)
        def st(name, arr):
            import numpy as _np
            a = _np.array(arr)
            print(f" {name}: shape={a.shape} min={a.min():.6e} max={a.max():.6e} mean={a.mean():.6e} std={a.std():.6e}")
        try:
            y1 = cunet.resnet_block_apply(params_ckpt['r1'], x2, t=t0)
            st('y1(after resnet)', y1)
            y1 = cunet.ws_conv_apply(params_ckpt['mix1'], y1)
            st('y1(after mix1)', y1)
            y1_att = cunet.self_attention_apply(params_ckpt['att1'], y1)
            st('y1(att out)', y1_att)
            y1 = y1 + params_ckpt.get('att_scale', 0.0) * y1_att
            st('y1(post-att)', y1)
            d1 = cunet.downsample(y1)
            st('d1(downsample)', d1)

            y2 = cunet.resnet_block_apply(params_ckpt['r2'], d1, t=t0)
            st('y2(after resnet)', y2)
            y2 = cunet.ws_conv_apply(params_ckpt['mix2'], y2)
            st('y2(after mix2)', y2)
            y2_att = cunet.self_attention_apply(params_ckpt['att2'], y2)
            st('y2(att out)', y2_att)
            y2 = y2 + params_ckpt.get('att_scale', 0.0) * y2_att
            st('y2(post-att)', y2)
            d2 = cunet.downsample(y2)
            st('d2(downsample)', d2)

            y3 = cunet.resnet_block_apply(params_ckpt['r3'], d2, t=t0)
            st('y3(after resnet)', y3)
            y3 = cunet.ws_conv_apply(params_ckpt['mix3'], y3)
            st('y3(after mix3)', y3)
            y3_att = cunet.self_attention_apply(params_ckpt['att3'], y3)
            st('y3(att out)', y3_att)
            y3 = y3 + params_ckpt.get('att_scale', 0.0) * y3_att
            st('y3(post-att)', y3)
            d3 = cunet.downsample(y3)
            st('d3(downsample)', d3)

            yb = cunet.resnet_block_apply(params_ckpt['rb'], d3, t=t0)
            st('yb(after rb)', yb)
            yb = cunet.ws_conv_apply(params_ckpt['mixb'], yb)
            st('yb(after mixb)', yb)

            u3 = cunet.upsample(yb)
            u3 = cunet.resize_to(u3, y3.shape[1], y3.shape[2])
            st('u3(before concat)', u3)
            u3 = jnp.concatenate([u3, y3], axis=-1)
            st('u3(after concat)', u3)
            u3 = cunet.ws_conv_apply(params_ckpt['proj3'], u3)
            st('u3(after proj3)', u3)
            u3 = cunet.resnet_block_apply(params_ckpt['d3'], u3, t=t0)
            st('u3(after d3 resnet)', u3)

            u2 = cunet.upsample(u3)
            u2 = cunet.resize_to(u2, y2.shape[1], y2.shape[2])
            u2 = jnp.concatenate([u2, y2], axis=-1)
            st('u2(after concat)', u2)
            u2 = cunet.ws_conv_apply(params_ckpt['proj2'], u2)
            st('u2(after proj2)', u2)
            u2 = cunet.resnet_block_apply(params_ckpt['d2'], u2, t=t0)
            st('u2(after d2 resnet)', u2)

            u1 = cunet.upsample(u2)
            u1 = cunet.resize_to(u1, y1.shape[1], y1.shape[2])
            u1 = jnp.concatenate([u1, y1], axis=-1)
            st('u1(after concat)', u1)
            u1 = cunet.ws_conv_apply(params_ckpt['proj1'], u1)
            st('u1(after proj1)', u1)
            u1 = cunet.resnet_block_apply(params_ckpt['d1'], u1, t=t0)
            st('u1(after d1 resnet)', u1)

            out2 = cunet.ws_conv_apply(params_ckpt['final'], u1)
            st('out2(final conv)', out2)
            out_c = out2[..., 0] + 1j * out2[..., 1]
            st('out_c(complex)', out_c)
        except Exception as e:
            print('Layer trace failed:', e)
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
        # If model was trained on [-1,1] scale, map to [0,1]; otherwise assume [0,1]
        mi = float(np.nanmin(imgs))
        ma = float(np.nanmax(imgs))
        if mi < -0.1 and ma <= 1.1:
            imgs = (imgs + 1.0) / 2.0
        # ensure valid image range
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
