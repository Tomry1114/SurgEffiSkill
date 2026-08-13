#!/bin/bash
#SBATCH --job-name=v3
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
cd "$HOME/hkust_surgery_video_learning"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 参数从**文件**读取。sbatch --export 按逗号切 KEY=VALUE，
# 而我们的参数里满是逗号（--ranks 0,2,3），此前一直在第一个逗号处被静默截断，
# 导致 --ranks 只剩 0、--epochs-grid/--lams 从未生效。
ARGS=$(cat "$ARGFILE")
echo "=== node=$(hostname) tag=$TAG ==="
echo "实际参数: $ARGS"
$HOME/.conda/envs/easyr1/bin/python -u "revision/scripts/sft_grpo/$SCRIPT" --tag "$TAG" $ARGS
echo "V3_DONE $TAG"
