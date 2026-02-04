#!/bin/bash
#SBATCH --job-name=ptycho_train
#SBATCH --output=/local/scratch/cfikes/PtyCoDiff2/logs/ptycho_train_%J.log
#SBATCH --error=/local/scratch/cfikes/PtyCoDiff2/logs/ptycho_train_%J.err
#SBATCH --account=mathg2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
##SBATCH --mail-type=ALL --mail-user=cfikes@emory.edu

set -euo pipefail

# Activate environment indirectly via conda run to avoid interactive shell issues

CONDA_PKG=/local/scratch/cfikes/miniconda3/bin/conda
ENV_PATH=/local/scratch/cfikes/PtyCoDiff2/conda-env

echo "Starting training job on $(hostname)"
echo "Conda env: ${ENV_PATH}"

# Ensure Python can import local packages
export PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2:${PYTHONPATH:-}

# Unbuffer stdout/stderr so logs show up immediately and enable JAX compile logs
export PYTHONUNBUFFERED=1
export JAX_LOG_COMPILES=1

# Exposed hyperparameters (can be overridden by environment or sbatch `--export`)
DATASET_DIR=${DATASET_DIR:-/local/scratch/cfikes/datasets/mnist}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-3}
GAMMA=${GAMMA:-0.99999}
EMA_DECAY=${EMA_DECAY:-0.99}
BETA=${BETA:-0.0}
MU_MAX=${MU_MAX:-10.0}
OUT_WEIGHTS=${OUT_WEIGHTS:-/local/scratch/cfikes/PtyCoDiff2/weights/no_attn3}
BASE_CH=${BASE_CH:-16}
MIXING=${MIXING:-0.0}
ATT_SCALE=${ATT_SCALE:-0}
N_T=${N_T:-128}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-10.0}
DIAGNOSTICS=${DIAGNOSTICS:-1}
#--lr 1e-4 --gamma 0.99999 --ema_decay 0.99 --beta 0.0 --mu_max 10.0 --base_ch 16 --mixing 0.0 --att_scale 0 --n_t 128 --out_weights /local/scratch/cfikes/PtyCoDiff2/weights/no_attn2 --max_grad_norm 10.0 --time_reweight_alpha 1.0 --time_reweight_lambda 3.0 --dc_weight 0 --b1 0.95 --b2 0.999 --diagnostics


# Time reweighting for DSM loss: weight(t) = 1 + alpha * exp(-lambda * t)
# Suggested: alpha >= 1.0 (makes earliest timesteps >=2x last timestep).
# Recommended starting values to test: alpha=1.0, lambda=5.0
TIME_REWEIGHT_ALPHA=${TIME_REWEIGHT_ALPHA:-1.0}
TIME_REWEIGHT_LAMBDA=${TIME_REWEIGHT_LAMBDA:-3.0}

# Small DC regularizer to discourage constant bias in eps predictions
DC_WEIGHT=${DC_WEIGHT:-0}
# Adam momentum (can tune to stabilize training)
B1=${B1:-0.95}
B2=${B2:-0.999}

PYTHON_BIN="${ENV_PATH}/bin/python"
if [ -x "${PYTHON_BIN}" ]; then
	PYTHON_CMD="${PYTHON_BIN} -u /local/scratch/cfikes/PtyCoDiff2/training/train.py --dataset_dir ${DATASET_DIR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} --gamma ${GAMMA} --ema_decay ${EMA_DECAY} --beta ${BETA} --mu_max ${MU_MAX} --base_ch ${BASE_CH} --mixing ${MIXING} --att_scale ${ATT_SCALE} --n_t ${N_T} --out_weights ${OUT_WEIGHTS} --max_grad_norm ${MAX_GRAD_NORM} --time_reweight_alpha ${TIME_REWEIGHT_ALPHA} --time_reweight_lambda ${TIME_REWEIGHT_LAMBDA} --dc_weight ${DC_WEIGHT} --b1 ${B1} --b2 ${B2}"
	if [ "${DIAGNOSTICS}" != "0" ]; then
		PYTHON_CMD="${PYTHON_CMD} --diagnostics"
	fi
else
	echo "Warning: ${PYTHON_BIN} not executable; falling back to conda run (may buffer logs)"
	PYTHON_CMD="${CONDA_PKG} run -p ${ENV_PATH} python -u /local/scratch/cfikes/PtyCoDiff2/training/train.py --dataset_dir ${DATASET_DIR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} --gamma ${GAMMA} --ema_decay ${EMA_DECAY} --beta ${BETA} --mu_max ${MU_MAX} --base_ch ${BASE_CH} --mixing ${MIXING} --att_scale ${ATT_SCALE} --n_t ${N_T} --out_weights ${OUT_WEIGHTS} --max_grad_norm ${MAX_GRAD_NORM} --time_reweight_alpha ${TIME_REWEIGHT_ALPHA} --time_reweight_lambda ${TIME_REWEIGHT_LAMBDA} --dc_weight ${DC_WEIGHT} --b1 ${B1} --b2 ${B2}"
	if [ "${DIAGNOSTICS}" != "0" ]; then
		PYTHON_CMD="${PYTHON_CMD} --diagnostics"
	fi
fi

echo "Running: ${PYTHON_CMD}"
eval ${PYTHON_CMD}

echo "Training job finished"
