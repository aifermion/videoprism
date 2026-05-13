#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name="videoprism-trial"
#SBATCH --time=01:40:00
#SBATCH --open-mode=append
#SBATCH --output=videoprism-trial-run.log
#SBATCH --error=trial-error.log
#SBATCH --gres=gpu:1

# ======== Job Execution Steps ========
cd /data/fbau775/videoprism

source /data/fbau775/miniconda3/bin/activate
conda activate videoprism

export HF_HOME=/data/fbau775/.cache/huggingface
export SSL_CERT_FILE=$CONDA_PREFIX/ssl/cacert.pem
export CURL_CA_BUNDLE=$CONDA_PREFIX/ssl/cacert.pem
export MPLCONFIGDIR=/data/fbau775/tmp/matplotlib
echo $HF_HOME

# Run your Python script or other commands
python zs-label-only.py \
    --csv test-mod-B.csv \
    --video_dir ../mammalps-dataset/benchmark_1/clips \
    --model videoprism_lvt_public_v1_base

