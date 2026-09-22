#!/usr/bin/env bash
# Experiment 2 (Sec. 4.1, Table 8): plant-based 70:20:10 split with the best model per line
# (Swin Transformer-B for Lines 1-2, ConvNeXt-B for Lines 3-4), followed by the
# spatio-temporal analysis of unseen test plants (Sec. 4.2, Figure 10).
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_table8.sh [extra --opts ...]
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A BEST=([1]=swin_b [2]=swin_b [3]=convnext_b [4]=convnext_b)
for LINE in 1 2 3 4; do
  MODEL=${BEST[$LINE]}
  echo ">>> Line ${LINE} | ${MODEL} | plant-based split"
  python main.py --line "${LINE}" --model "${MODEL}" --split plant "$@"
  RUN=$(ls -td runs/line${LINE}_${MODEL}_plant* | head -1)
  python scripts/temporal_monitoring.py \
    --predictions "${RUN}/eval/predictions_test.csv" \
    --classes "data/splits/plant/line${LINE}/classes.json"
done

python scripts/collect_results.py --split plant
