#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name="vp-bench1-test"
#SBATCH --time=08:00:00
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


# ======== Test-set Evaluation ========
echo ""
echo "========================================"
echo "  Testing with seed=42"
echo "========================================"
python train_benchmark1.py test \
    --data_dir ../mammalps-dataset/benchmark_1 \
    --model_size base \
    --batch_size 16 \
    --num_workers 4 \
    --ckpt_dir checkpoints/benchmark1_finetune \
    --output_dir "results/benchmark1/test_rerun_42" \
    --seed 42

echo "Job finished at $(date)"

