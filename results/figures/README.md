# Figures

| Paper figure | Status in this repository | How to regenerate after training |
|---|---|---|
| Fig. 1–4 (strategy, sample images, health levels, framework diagram) | Not included. These are photographs and diagrams from the paper, not data plots | — |
| Fig. 5 Confusion matrices | `fig5_confusion_matrices_reported.png`, **redrawn from the values printed in the paper** | `python scripts/evaluate.py ... --plot` |
| Fig. 6 t-SNE | Code only. Needs the trained models' features | `python scripts/tsne.py ...` |
| Fig. 7 Grad-CAM | Code only. Needs trained models and images | `python scripts/gradcam.py ... --select correct` |
| Fig. 8 Misclassification cases | Code only | `python scripts/gradcam.py ... --select misclassified` |
| Fig. 9 Qualitative predictions | Code only | `python scripts/qualitative_examples.py --predictions ...` |
| Fig. 10 Predicted vs expert state over time | Code only. Needs plant-split predictions | `python scripts/temporal_monitoring.py ...` |
| Fig. 11 Expert health trajectories | Code only. Needs `data/annotations.csv` | `python scripts/plot_expert_trajectories.py` |

No data points were invented to recreate Figures 6–11. Figures produced by the scripts are written
next to the run (`runs/<experiment>/eval/`) or to `results/figures/generated/`.
