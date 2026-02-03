#!/usr/bin/env python3
"""
train.py: Main training script for complex UNet score model (ptychographic generative modeling)
- Data/model loading based on train_mnist_stripped.py
- CLI and hyperparameters as in run_train.sbatch
- Gradient clipping always enabled
- If debug=0, only loss is computed (no diagnostics/aux metrics)
- If debug!=0, compute all metrics as in ptycho_train_138171.log
"""
import os

import os
import sys
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root on path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


from diffusion.model import create_complexUnet

def cosine_alpha_sigma(t):
    # VP SDE: alpha(t), sigma(t) for cosine schedule
    s = 0.008
    t_ = jnp.clip(t, 0.0, 1.0)
    f = jnp.cos((t_ + s) / (1 + s) * jnp.pi / 2)
    alpha = f
    sigma = jnp.sqrt(1 - f ** 2)
    return alpha, sigma

def adam_init(params):
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    return m, v

def adam_update(params, grads, m, v, step, lr=1e-4, max_grad_norm=1.0, b1=0.95, b2=0.999, eps=1e-8):
    # Adam optimizer with gradient clipping
    gnorm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    clip_coef = jnp.minimum(1.0, max_grad_norm / (gnorm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * clip_coef, grads)
    m = jax.tree_util.tree_map(lambda m, g: (1 - b1) * g + b1 * m, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: (1 - b2) * (g ** 2) + b2 * v, v, grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** (step + 1)), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** (step + 1)), v)
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, mhat, vhat)
    return params, m, v, gnorm, clip_coef

def sanitize_params(params):
    # Convert all int dtypes to float32 for JAX compatibility
    def to_float(x):
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(to_float, params)

def sample_cn(key, shape, dtype=jnp.complex64):
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, shape, dtype=jnp.float32)
    im = jax.random.normal(ki, shape, dtype=jnp.float32)
    return ((re + 1j * im) / jnp.sqrt(2.0)).astype(dtype)

def save_pytree_npz(path, params, ema_params, meta_dict):
    leaves_p, treedef_p = jax.tree_util.tree_flatten(params)
    leaves_p = [np.array(x) for x in leaves_p]
    leaves_e, treedef_e = jax.tree_util.tree_flatten(ema_params)
    leaves_e = [np.array(x) for x in leaves_e]
    treedef_p_bytes = np.frombuffer(pickle.dumps(treedef_p), dtype=np.uint8)
    treedef_e_bytes = np.frombuffer(pickle.dumps(treedef_e), dtype=np.uint8)
    meta_bytes = np.frombuffer(pickle.dumps(meta_dict), dtype=np.uint8)
    save_dict = {
        'treedef_params': treedef_p_bytes,
        'treedef_ema': treedef_e_bytes,
        'metadata': meta_bytes,
    }
    for i, arr in enumerate(leaves_p):
        save_dict[f'params_{i:04d}'] = arr
    for i, arr in enumerate(leaves_e):
        save_dict[f'ema_{i:04d}'] = arr
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    np.savez(path, **save_dict)

def load_mnist_npz(path):
    data = np.load(path)
    if 'images' in data:
        images = data['images']
    elif 'x' in data:
        images = data['x']
    elif 'arr_0' in data:
        images = data['arr_0']
    else:
        keys = list(data.keys())
        images = data[keys[0]]
    return images, data.get('labels_real', None), data.get('labels_imag', None)

def main():
    parser = argparse.ArgumentParser(description='Complex UNet score model training')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--gamma', type=float, default=0.9999)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--mu_max', type=float, default=10.0)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--mixing', type=float, default=0.01)
    parser.add_argument('--att_scale', type=float, default=0)
    parser.add_argument('--n_t', type=int, default=128)
    parser.add_argument('--out_weights', type=str, required=True)
    parser.add_argument('--max_grad_norm', type=float, default=1000.0)
    parser.add_argument('--time_reweight_alpha', type=float, default=1.0)
    parser.add_argument('--time_reweight_lambda', type=float, default=3.0)
    parser.add_argument('--dc_weight', type=float, default=0)
    parser.add_argument('--b1', type=float, default=0.95)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()

    images_np, _, _ = load_mnist_npz(os.path.join(args.dataset_dir, 'train.npz'))
    images = jnp.array(images_np.astype(np.complex64))
    N = images.shape[0]
    print(f'Loaded {N} images, shape={images.shape}')

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    params, apply_fn = create_complexUnet(
        subkey,
        input_shape=(28, 28, 1),
        base_ch=args.base_ch,
        mixing=args.mixing
    )
    params = sanitize_params(params)
    m, v = adam_init(params)
    ema_params = params
    t_grid = jnp.linspace(0.0, 1.0, args.n_t + 1)

    steps_per_epoch = N // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    print(f'Starting training for {args.epochs} epochs ({total_steps} steps)...')
    t0 = time.time()
    for step in range(total_steps):
        key, subkey = jax.random.split(key)
        batch_indices = jax.random.randint(subkey, (args.batch_size,), 0, N)
        key, subkey = jax.random.split(key)
        n_steps = t_grid.shape[0] - 1
        t_idx = jax.random.randint(subkey, (args.batch_size,), 1, n_steps + 1)
        t = t_grid[t_idx]
        a, s = cosine_alpha_sigma(t)
        a = a.reshape((args.batch_size, 1, 1, 1))
        s = s.reshape((args.batch_size, 1, 1, 1))
        key, subkey = jax.random.split(key)
        batch = images[batch_indices]
        eps = sample_cn(subkey, batch.shape)
        x_t = a * batch + s * eps

        def loss_fn(params):
            eps_pred = apply_fn(params, x_t, t)
            # Score loss (per-pixel MSE)
            err = jnp.sum(jnp.abs(eps_pred - eps) ** 2, axis=(1, 2, 3))
            n_pix = jnp.asarray(batch.shape[1] * batch.shape[2] * batch.shape[3], dtype=jnp.float32)
            score_loss = jnp.mean(err / (n_pix + 1e-12))

            # Gauge loss (Tweedie denoising + phase alignment)
            if args.beta > 0:
                # Tweedie denoised estimate
                x0_hat = (x_t + (s ** 2) * eps_pred) / (a + 1e-8)
                # Compute optimal global phase for each sample
                def phi_star(x0h, x0):
                    ip = jnp.vdot(x0h, x0)
                    norm = jnp.linalg.norm(x0h) * jnp.linalg.norm(x0) + 1e-8
                    return jnp.angle(ip / norm)
                phi = jax.vmap(phi_star)(x0_hat.reshape(args.batch_size, -1), batch.reshape(args.batch_size, -1))
                # SNR-weighted schedule
                mu = jnp.clip((a ** 2) / (s ** 2 + 1e-8), 0, args.mu_max)
                mu = mu.reshape(-1)
                gauge_loss = jnp.mean(mu * (1.0 - jnp.cos(phi)))
            else:
                gauge_loss = 0.0

            return score_loss + args.beta * gauge_loss, (score_loss, gauge_loss)

        (loss_val, (score_loss_val, gauge_loss_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        params, m, v, gnorm, gcoef = adam_update(
            params, grads, m, v, step,
            lr=args.lr, max_grad_norm=args.max_grad_norm, b1=args.b1, b2=args.b2
        )
        ema_params = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            ema_params, params
        )
        # Print diagnostics if debug!=0 every 10 steps, else only at epoch end
        if args.debug != 0 and (step % 10 == 0 or (step + 1) % steps_per_epoch == 0):
            print(f"Step {step} loss={float(loss_val):.6e} score={float(score_loss_val):.6e} gauge={float(gauge_loss_val):.6e} grad_norm={float(gnorm):.3e} clip_coef={float(gcoef):.3e}")
        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            # Compute diagnostics for this epoch
            if args.debug != 0:
                # Use a fixed batch for diagnostics
                key_diag = jax.random.PRNGKey(epoch)
                batch_indices = jax.random.randint(key_diag, (args.batch_size,), 0, N)
                batch = images[batch_indices]
                t_diag = t_grid[jax.random.randint(key_diag, (args.batch_size,), 1, t_grid.shape[0])]
                a_diag, s_diag = cosine_alpha_sigma(t_diag)
                a_diag = a_diag.reshape((args.batch_size, 1, 1, 1))
                s_diag = s_diag.reshape((args.batch_size, 1, 1, 1))
                key_eps = jax.random.PRNGKey(epoch + 9999)
                eps_diag = sample_cn(key_eps, batch.shape)
                x_t_diag = a_diag * batch + s_diag * eps_diag
                eps_pred_diag = apply_fn(params, x_t_diag, t_diag)
                mean_eps_pred = float(jnp.mean(jnp.abs(eps_pred_diag)))
                mean_eps = float(jnp.mean(jnp.abs(eps_diag)))
                implied_score_mag = float(jnp.mean(jnp.abs(eps_pred_diag) / (jnp.maximum(s_diag, 1e-6))))
                # Compute param norm
                def tree_norm(tree):
                    leaves = jax.tree_util.tree_leaves(tree)
                    return float(jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves])))
                param_norm = tree_norm(params)
                print(f"Step {step} mean|eps_pred|={mean_eps_pred:.6e} mean|eps|={mean_eps:.6e} implied_score_mag={implied_score_mag:.6e}")
                print(f"Epoch {epoch}/{args.epochs} loss={float(loss_val):.6e} time={time.time()-t0:.2f}s grad_norm={float(gnorm):.6e} param_norm={param_norm:.6e}")
            else:
                print(f"Epoch {epoch}/{args.epochs} loss={float(loss_val):.6e} score={float(score_loss_val):.6e} gauge={float(gauge_loss_val):.6e} time={time.time()-t0:.2f}s grad_norm={float(gnorm):.3e}")
            t0 = time.time()
    meta_dict = vars(args)
    save_pytree_npz(args.out_weights, params, ema_params, meta_dict)
    print('Training complete!')

if __name__ == '__main__':
    main()
    # gauge_loss = 0.0

    # # return score_loss + args.beta * gauge_loss, (score_loss, gauge_loss)

    # (loss_val, (score_loss_val, gauge_loss_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # params, m, v, gnorm, gcoef = train_mod.adam_update(
    #     params, grads, m, v, step,
    #     lr=args.lr, max_grad_norm=args.max_grad_norm
    # )
    # ema_params = jax.tree_util.tree_map(
    #     lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
    #     ema_params, params
    # )
    # if args.debug != 0 or args.diagnostics:
    #     print(f"Step {step} loss={float(loss_val):.6e} score={float(score_loss_val):.6e} gauge={float(gauge_loss_val):.6e} grad_norm={float(gnorm):.3e} clip_coef={float(gcoef):.3e}")
    # if (step + 1) % steps_per_epoch == 0:
    #     epoch = (step + 1) // steps_per_epoch
    #     print(f"Epoch {epoch}/{args.epochs} loss={float(loss_val):.6e} score={float(score_loss_val):.6e} gauge={float(gauge_loss_val):.6e} time={time.time()-t0:.2f}s grad_norm={float(gnorm):.3e}")
    #     t0 = time.time()
    # meta_dict = vars(args)
    # save_pytree_npz(args.out_weights, params, ema_params, meta_dict)
    # print('Training complete!')

# if __name__ == '__main__':
#     main()
#     N = images.shape[0]
#     print(f'Loaded {N} images, shape={images.shape}')
#     images = jnp.array(images_np.astype(np.complex64))
#     N = images.shape[0]
#     print(f'Loaded {N} images, shape={images.shape}')
#     x_t = a * batch + s * eps

#     loss_val, grads = jax.value_and_grad(loss_fn)(params)
#     params, m, v, gnorm, gcoef = train_mod.adam_update(
#         params, grads, m, v, step,
#         lr=args.lr, max_grad_norm=args.max_grad_norm
#     )
#     ema_params = jax.tree_util.tree_map(
#         lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
#         ema_params, params
#     )
#     if args.debug != 0 or args.diagnostics:
#         # Compute and print diagnostics as in log
#         eps_pred = apply_fn(params, x_t, t)
#         mean_eps_pred = float(jnp.mean(jnp.abs(eps_pred)))
#         mean_eps = float(jnp.mean(jnp.abs(eps)))
#         print(f'Step {step} grad_norm={gnorm:.6e} clip={gcoef:.3e} lr={args.lr:.3e}')
#         print(f'Step {step} mean|eps_pred|={mean_eps_pred:.6e} mean|eps|={mean_eps:.6e}')
#     if (step + 1) % steps_per_epoch == 0:
#         epoch = (step + 1) // steps_per_epoch
#         print(f'Epoch {epoch}/{args.epochs} loss={float(loss_val):.6e} time={time.time()-t0:.2f}s grad_norm={gnorm:.6e}')
#         t0 = time.time()
#     meta_dict = vars(args)
#     save_pytree_npz(args.out_weights, params, ema_params, meta_dict)
#     print('Training complete!')

# if __name__ == '__main__':
#     main()
