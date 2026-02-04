import pickle
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from diffusion.sampler import make_sampler
from diffusion.model import create_complexUnet, load_checkpoint_params

CKPT = '/local/scratch/cfikes/PtyCoDiff2/weights/no_attn3.ckpt_100.npz'
OUT = '/local/scratch/cfikes/PtyCoDiff2/weights/no_attn/sampled_images.png'
P = 16

print('Loading checkpoint', CKPT)
params_ckpt, apply_fn = load_checkpoint_params(CKPT, jax.random.PRNGKey(0), input_shape=(28,28,1))

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

# For visualization use real part. Map to [0,1] only if in [-1,1].
imgs = np.array(jnp.real(samples))
mi = float(np.nanmin(imgs))
ma = float(np.nanmax(imgs))
if mi < -0.1 and ma <= 1.1:
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
