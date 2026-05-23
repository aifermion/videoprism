#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name="vp-bench1-rerun"
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=benchmark1-rerun-%j.log
#SBATCH --error=benchmark1-rerun-%j.err
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1

# ======== Environment Setup ========
cd /data/fbau775/videoprism

source /data/fbau775/miniconda3/bin/activate
conda activate videoprism

export HF_HOME=/data/fbau775/.cache/huggingface
export SSL_CERT_FILE=$CONDA_PREFIX/ssl/cacert.pem
export CURL_CA_BUNDLE=$CONDA_PREFIX/ssl/cacert.pem
export MPLCONFIGDIR=/data/fbau775/tmp/matplotlib

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ======== Abort early if no GPU ========
python -c "import jax; assert jax.default_backend() != 'cpu', 'No GPU detected — aborting to avoid slow CPU training'" || exit 1

# ======== Training + Evaluation for multiple seeds ========
SEEDS=(42 163 689)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Training with seed=${SEED}"
    echo "========================================"
    python train_benchmark1_RERUN.py train \
        --data_dir ../mammalps-dataset/benchmark_1 \
        --model_size base \
        --num_epochs 30 \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_rerun_seed_${SEED}" \
        --ckpt_every 50 \
        --keep_recent 5 \
        --output_dir "results/benchmark1_rerun/train_seed_${SEED}" \
        --seed "$SEED"

    echo ""
    echo "========================================"
    echo "  Evaluating with seed=${SEED}"
    echo "========================================"
    python train_benchmark1_RERUN.py test \
        --data_dir ../mammalps-dataset/benchmark_1 \
        --model_size base \
        --batch_size 16 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_rerun_seed_${SEED}" \
        --output_dir "results/benchmark1_rerun/test_seed_${SEED}" \
        --seed "$SEED"
done

echo ""
echo "========================================"
echo "  All seeds complete"
echo "========================================"
echo "Job finished at $(date)"


## Call old training script

