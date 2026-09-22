# Results

> **Reported results from the paper — not independently reproduced by this repository.**
>
> Every number in `paper_results/` and `tables/` was transcribed from Fuentes, Asgher et al. (2025),
> *Front. Plant Sci.* 16:1511651. None of them was produced by running the code in this repository.

| Folder / file | Content | Paper source |
|---|---|---|
| `paper_results/reported_results.json` | Single source of truth for all reported numbers | Tables 1, 3–8, Figure 5, Sec. 4.2 |
| `tables/table1_data_acquisition.{csv,tex}` | Lines, varieties, periods, weeks, plants | Table 1 |
| `tables/table3_class_distribution.{csv,tex}` | Images per health level and line | Table 3 |
| `tables/table4_split_sizes.{csv,tex}` | Train/validation sizes (80:20) | Table 4 |
| `tables/table6_main_results.{csv,tex}` | 5 models × 4 lines: train/val accuracy, precision, recall, F1 | Table 6 |
| `tables/table7_classwise_results.{csv,tex}` | Per-class precision/recall/F1 of the best model per line | Table 7 |
| `tables/table8_plant_based_results.{csv,tex}` | Plant-based 70:20:10 split: train/val/test accuracy | Table 8 |
| `tables/figure5_confusion_matrices.{csv,tex}` | Row-normalised confusion matrices, as printed in the figure | Figure 5 |
| `figures/fig5_confusion_matrices_reported.png` | Figure 5 redrawn from the printed cell values | Figure 5 |
| `reproduced/` | Empty. `scripts/collect_results.py` writes your own re-run results here | — |

The tables are generated from the JSON file with `python scripts/export_reported_tables.py`.
That script only reformats the paper's numbers.

The CSV files begin with two `#` comment lines. Read them with `pd.read_csv(path, comment='#')`.
