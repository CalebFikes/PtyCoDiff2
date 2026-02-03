import pickle
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from diffusion.sampler import make_sampler
from diffusion.model import create_complexUnet

CKPT = '/local/scratch/cfikes/PtyCoDiff2/weights/no_attn/ckpt_020.pkl'
OUT = '/local/scratch/cfikes/PtyCoDiff2/weights/no_attn/sampled_images.png'
P = 16

print('Loading checkpoint', CKPT)
with open(CKPT, 'rb') as f:
    d = pickle.load(f)

params_ckpt = d.get('ema_params', d.get('params'))
if params_ckpt is None:
    raise SystemExit('No params found in checkpoint')

# Recreate an apply_fn with statics that match the checkpoint when possible.
# Infer `base_ch`, `att_scale`, and `mixing` from the saved params to avoid
# architecture mismatches when applying the checkpoint.
rng = jax.random.PRNGKey(0)
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
    # fetch stored att_scale if present
    try:
        att_scale = float(params_ckpt.get('att_scale', att_scale))
    except Exception:
        pass
    # mixing module may store a scalar mix weight under 'm1' etc.
    try:
        mixing = float(params_ckpt.get('m1', {}).get('mix', mixing))
    except Exception:
        pass
_, apply_fn = create_complexUnet(rng, input_shape=(28,28,1,1), base_ch=base_ch, mixing=mixing, att_scale=att_scale)

cfg = {'n_t': 128, 'object_shape': (28,28,1), 'predicts_eps': True}
print('Making sampler')
sampler = make_sampler(cfg, None, lambda p, x, t: apply_fn(p, x, t))
# If the returned sampler is jitted, call the original Python implementation
# to avoid traced/dynamic-shape issues when P is a runtime argument.
sampler_fn = getattr(sampler, '__wrapped__', sampler)

key = jax.random.PRNGKey(42)
print('Sampling')
samples = sampler_fn(params_ckpt, key, P=P)
# samples: (P,H,W,C) complex

# For visualization use real part and rescale from [-1,1] -> [0,1]
imgs = np.array(jnp.real(samples))
imgs = (imgs + 1.0) / 2.0
imgs = np.clip(imgs, 0.0, 1.0)

ncol = 4
nrow = int(np.ceil(P / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
axes = axes.flatten()
for i in range(nrow * ncol):
    ax = axes[i]
    ax.axis('off')
    if i < P:
        im = imgs[i,...,0]
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print('Saved samples to', OUT)
