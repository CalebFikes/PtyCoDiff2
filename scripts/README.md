# Stripped Training & Sampling Scripts

Minimal, reproducible training and sampling for complex MNIST diffusion.

## Overview

- **train_mnist_stripped.py**: Trains a ComplexUNetV2 model using eps-prediction with discrete timestep sampling
- **sample_debug_sampling.py**: Samples from trained model using Annealed Langevin Dynamics (ALD)

## Quick Start

### Training

```bash
python scripts/train_mnist_stripped.py \
    --mnist_path /local/scratch/cfikes/datasets/mnist/train.npz \
    --steps 10000 \
    --batch_size 512 \
    --lr 1e-4 \
    --out_path weights/debug_sampling.npz
```

**Key hyperparameters** (matching successful test runs):
- `--base_ch 8`: Smaller model for faster training
- `--mixing 0.0`: Attention disabled for simplicity
- `--lr 1e-4`: Learning rate (default)
- `--max_grad_norm 1.0`: Gradient clipping (default)
- `--n_t 128`: Number of discrete timesteps (default)

### Sampling

```bash
python scripts/sample_debug_sampling.py \
    --weights_path weights/debug_sampling.npz \
    --n_samples 8 \
    --n_steps 128 \
    --use_ema
```

Generates a 4×4 grid showing real (top 2 rows) and imaginary (bottom 2 rows) parts of 8 samples.

## Implementation Details

### Training Convention
- **Noise prediction**: Model outputs `eps_pred ≈ eps` where `eps ~ CN(0,I)`
- **VP schedule**: `alpha(t) = cos(π/2 * t)`, `sigma(t) = sin(π/2 * t)`
- **Forward process**: `x_t = alpha(t) * x0 + sigma(t) * eps`
- **Loss**: Per-pixel MSE: `mean(|eps_pred - eps|²) / n_pixels`

### Sampling (ALD)
- **Score conversion**: `score(x_t, t) = -eps_pred / sigma(t)`
- **Step size**: `gamma_k = eps_step * (sigma_k / sigma_max)²`
- **Update**: `x_{k+1} = x_k + gamma_k * score + sqrt(2 * gamma_k) * zeta`
- Descending time schedule from `t=1.0` to `t=0.0`

### Data Format
- MNIST preprocessed to complex images with real/imag ∈ [-1,1]
- Weights saved as `.npz` with pytree serialization (flattened leaves + pickled treedef)
- Metadata includes: `base_ch`, `mixing`, `n_t`, `steps`, `lr`, etc.

## Test Results

Training on 500 steps (batch_size=128):
```
Step     0 | loss=9.93e-01 | mean|eps_pred|=1.76e+01 | mean|eps|=8.84e-01
Step   200 | loss=1.08e+00 | mean|eps_pred|=2.26e-01 | mean|eps|=8.85e-01
Step   400 | loss=7.47e-01 | mean|eps_pred|=3.73e-01 | mean|eps|=8.88e-01
Final loss: 6.20e-01
```

Model learns to reduce `mean|eps_pred|` from 17.6 → 0.37 (approaching target ~0.88).

## Dependencies

- JAX (with CUDA support recommended)
- NumPy
- matplotlib (for sampling visualization)
- tqdm (optional, for training progress)

## File Structure

```
scripts/
├── train_mnist_stripped.py    # Training script
└── sample_debug_sampling.py   # Sampling script

weights/
├── debug_sampling.npz          # Trained model weights
└── debug_sampling_grid.png     # Sample outputs (4×4 grid)
```
