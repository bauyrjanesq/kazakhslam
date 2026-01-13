#!/bin/bash
#SBATCH --job-name=slam-kazakh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -----------------------------
# Environment setup
# -----------------------------
set -e
set -o pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
conda activate myenv   # or source venv/bin/activate

# Optional: limit CPU threading
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Training command
# -----------------------------
python cli/train.py \
  data.train_path=data/kz_tokens_train_real.jsonl \
  data.val_path=data/kz_tokens_dev.jsonl \
  model=slam \
  training_args.per_device_train_batch_size=8 \
  training_args.gradient_accumulation_steps=16 \
  training_args.output_dir=pretrained_unitlm_kazakh \
  training_args.bf16=false \
  +training_args.max_steps=17625

echo "Job finished on $(date)"