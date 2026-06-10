#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name="mammalps-rerun"
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=benchmark1-train-%j.log
#SBATCH --error=benchmark1-train-%j.err
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
SEEDS=(42 163)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Training with seed=${SEED}"
    echo "  Starting at $(date)"
    echo "========================================"
    python train_benchmark1.py train \
        --data_dir ../mammalps-dataset/benchmark_1 \
        --model_size base \
        --num_epochs 150 \
        --batch_size 16 \
        --learning_rate 1e-5 \
        --min_learning_rate 1e-7 \
        --weight_decay 0.01 \
        --dropout 0.1 \
        --head_hidden_dim 256 \
        --loss_weight_actions 2.0 \
        --loss_weight_species 1.0 \
        --loss_weight_activity 2.5 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_RERUN_finetune_seed_${SEED}" \
        --ckpt_every 50 \
        --keep_recent 5 \
        --output_dir "results/benchmark1/train_seed_${SEED}" \
        --seed "$SEED"

    echo ""
    echo "========================================"
    echo "  Evaluating with seed=${SEED}"
    echo "========================================"
    python train_benchmark1.py test \
        --data_dir ../mammalps-dataset/benchmark_1 \
        --model_size base \
        --head_hidden_dim 256 \
        --batch_size 16 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_RERUN_finetune_seed_${SEED}" \
        --output_dir "results/benchmark1/test_seed_${SEED}" \
        --seed "$SEED"

    echo ""
    echo "========================================"
    echo "  Multi-sample evaluating with seed=${SEED}"
    echo "========================================"
    python train_benchmark1.py test_ms \
        --data_dir ../mammalps-dataset/benchmark_1 \
        --model_size base \
        --head_hidden_dim 256 \
        --batch_size 16 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_RERUN_finetune_seed_${SEED}" \
        --output_dir "results/benchmark1/test_seed_${SEED}" \
        --num_test_samples 10 \
        --min_clip_duration 0.5 \
        --seed "$SEED"

    echo "========================================"
    echo "Seed ${SEED} finished at $(date)"
    echo "========================================"
done

echo "Job finished at $(date)"
