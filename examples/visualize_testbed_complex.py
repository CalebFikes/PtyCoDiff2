"""
Visualize complex testbed outputs saved by `run_conditional_viz.py`.

Generates:
 - measurements grid image
 - top-8 reconstructions (real and imag) as PNGs

Usage:
  python examples/visualize_testbed_complex.py runs/posterior_data.npz --outdir runs/posterior_viz
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize(npz_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    # expected keys: 'particles' (n, H, W, C) complex, 'weights' (n,), 'measurements' (n_meas, ph, pw), 'positions' (n_meas,2)
    particles = data['particles']
    if particles.dtype.kind == 'c':
        pass
    else:
        # allow real by viewing as complex
        particles = particles.astype(np.complex64)

    weights = data.get('weights', np.ones(particles.shape[0]))
    measurements = data['measurements']
    positions = data['positions']

    n, H, W, C = particles.shape

    # Measurements montage
    n_meas = measurements.shape[0]
    cols = min(8, n_meas)
    rows = (n_meas + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i in range(len(axes)):
        ax = axes[i]
        if i < n_meas:
            im = measurements[i]
            im_norm = im / (np.max(im) + 1e-10)
            ax.imshow(1.0 - im_norm, cmap='gray')
            ax.set_title(f"Meas {i}\nPos {int(positions[i,0])},{int(positions[i,1])}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(outdir / 'measurements_montage.png', dpi=150)
    plt.close()

    # Top reconstructions (up to 8)
    top_k = min(8, particles.shape[0])
    idxs = np.argsort(weights)[-top_k:][::-1]
    for rank, idx in enumerate(idxs):
        p = particles[idx]
        # Real
        fig, ax = plt.subplots(1,1, figsize=(3,3))
        real = np.real(p[...,0]) if p.ndim == 3 else np.real(p)
        real_norm = (real + 1.0) / 2.0
        ax.imshow(real_norm, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Top{rank+1} Real (w={weights[idx]:.3f})')
        ax.axis('off')
        plt.savefig(outdir / f'top{rank+1}_real.png', dpi=150)
        plt.close()

        # Imag (magnitude)
        fig, ax = plt.subplots(1,1, figsize=(3,3))
        imag = np.abs(np.imag(p[...,0]) if p.ndim == 3 else np.abs(np.imag(p)))
        imag_norm = imag / (np.max(imag) + 1e-10)
        ax.imshow(imag_norm, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Top{rank+1} Imag')
        ax.axis('off')
        plt.savefig(outdir / f'top{rank+1}_imag.png', dpi=150)
        plt.close()

    print(f"Saved visualizations to {outdir}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('npz', type=str)
    p.add_argument('--outdir', type=str, default=None)
    args = p.parse_args()
    npz = Path(args.npz)
    outdir = args.outdir or npz.parent
    visualize(npz, outdir)
