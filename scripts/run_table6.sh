#!/usr/bin/env bash
# Experiment 1 (Sec. 3.2, Tables 6-7, Figures 5-9):
# 5 architectures x 4 cultivation lines, image-level random 80:20 split.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_table6.sh [extra --opts ...]
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(vgg16 resnet18 swin_b vit_b convnext_b)
for LINE in 1 2 3 4; do
  for MODEL in "${MODELS[@]}"; do
    echo ">>> Line ${LINE} | ${MODEL} | random split"
    python main.py --line "${LINE}" --model "${MODEL}" --split random "$@"
  done
done

python scripts/collect_results.py --split random
