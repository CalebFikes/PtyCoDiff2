#!/bin/bash
#SBATCH --job-name=train_mnist_stripped
#SBATCH --output=logs/train_mnist_stripped.%j.out
#SBATCH --error=logs/train_mnist_stripped.%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
echo "Running train_mnist_stripped in $PWD"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
mkdir -p logs weights
export PYTHONUNBUFFERED=1

# Use conda env python with CUDA-enabled jaxlib
PYTHON_BIN=/local/scratch/cfikes/PtyCoDiff2/conda-env/bin/python
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=/local/scratch/cfikes/miniconda3/bin/python
fi

# Prefer CUDA only when available
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export JAX_PLATFORMS=cpu
else
    export JAX_PLATFORMS=cuda
fi

# Compute steps for 50 epochs with batch_size=512
# 60000 images / 512 = 117.2 steps per epoch -> 117 steps/epoch
# 50 epochs * 117 = 5850 steps
STEPS=5850

echo "Training for $STEPS steps (50 epochs, batch_size=512)"

$PYTHON_BIN -u scripts/train_mnist_stripped.py \
    --mnist_path /local/scratch/cfikes/datasets/mnist/train.npz \
    --out_path weights/mnist_stripped_50ep.npz \
    --steps $STEPS \
    --batch_size 512 \
    --base_ch 8 \
    --mixing 0.0 \
    --lr 1e-4 \
    --max_grad_norm 1.0 \
    --n_t 128 \
    --ema_decay 0.999 \
    --seed 42

echo "Training complete!"
