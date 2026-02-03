#!/usr/bin/env python3
"""Minimal sampling script using ALD (Annealed Langevin Dynamics).

Loads weights from .npz and generates complex image samples.
"""
import os
import sys
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure repo root on path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from training import train as train_mod
from diffusion.model import create_complexUnet


def sample_cn(key, shape, dtype=jnp.complex64):
    """Sample complex Gaussian CN(0,I): real/imag ~ N(0, 1/2)."""
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, shape, dtype=jnp.float32)
    im = jax.random.normal(ki, shape, dtype=jnp.float32)
    return ((re + 1j * im) / jnp.sqrt(2.0)).astype(dtype)


def load_pytree_npz(path):
    """Load params and ema_params from .npz."""
    data = np.load(path)
    
    # Deserialize metadata
    meta_dict = pickle.loads(data['metadata'].tobytes())
    
    # Deserialize treedefs
    treedef_p = pickle.loads(data['treedef_params'].tobytes())
    treedef_e = pickle.loads(data['treedef_ema'].tobytes())
    
    # Reconstruct params
    leaves_p = []
    i = 0
    while f'params_{i:04d}' in data:
        leaves_p.append(jnp.array(data[f'params_{i:04d}']))
        i += 1
    #!/usr/bin/env python3
    """Obsolete sampling script stub.

    This script was replaced by the central sampler implementation in
    `diffusion/sampler.py`. Import and use `make_sampler` from that module.
    """

    raise RuntimeError(
        "Deprecated: use diffusion.sampler.make_sampler(...) instead of this script."
    )
            ax = axes[i, j]
            
            # First 2 rows: real parts (samples 0-7)
            if i < 2:
                sample_idx = i * 4 + j
                if sample_idx < args.n_samples:
                    img = samples_real[sample_idx, :, :, 0]
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'Real {sample_idx}', fontsize=8)
                    ax.axis('off')
                else:
                    ax.axis('off')
            # Last 2 rows: imag parts (samples 0-7)
            else:
                sample_idx = (i - 2) * 4 + j
                if sample_idx < args.n_samples:
                    img = samples_imag[sample_idx, :, :, 0]
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'Imag {sample_idx}', fontsize=8)
                    ax.axis('off')
                else:
                    ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(args.plot_path) if os.path.dirname(args.plot_path) else '.', exist_ok=True)
    plt.savefig(args.plot_path, dpi=150, bbox_inches='tight')
    print(f'Saved samples to {args.plot_path}')


if __name__ == '__main__':
    main()
