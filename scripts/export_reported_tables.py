"""Write the paper's reported numbers (results/paper_results/reported_results.json) as
CSV and LaTeX tables in results/tables/ and redraw Figure 5 from the values printed in
the paper's confusion matrices.

This script only re-formats numbers that are already in the paper. It trains nothing
and computes no new results.

Usage:
    python scripts/export_reported_tables.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from plant_health.utils.config import REPO_ROOT

NOTICE = 'Reported results from the paper - not independently reproduced by this repository.'
TABLES = os.path.join(REPO_ROOT, 'results', 'tables')
FIGURES = os.path.join(REPO_ROOT, 'results', 'figures')


def fmt(v, column=''):
    """Print numbers the way the paper prints them: accuracies with 3 decimals, precision /
    recall / F1 with 2 decimals, more digits only when the paper gives more (e.g. 0.6498)."""
    if not isinstance(v, float):
        return str(v)
    given = len(repr(v).split('.')[-1])
    if 'Accuracy' in column:
        return '{:.{}f}'.format(v, max(3, given))
    if column in ('Precision', 'Recall', 'F1-Score'):
        return '{:.{}f}'.format(v, max(2, given))
    return repr(v)


def formatted(df):
    out = df.copy().astype(object)
    for c in out.columns:
        out[c] = [fmt(v, c) for v in df[c]]
    return out


def write(df, name, caption, bold_rows=None):
    df = formatted(df)
    csv_path = os.path.join(TABLES, name + '.csv')
    with open(csv_path, 'w') as f:
        f.write('# {}\n# {}\n'.format(caption, NOTICE))
        df.to_csv(f, index=False)

    cols = list(df.columns)
    lines = ['% ' + NOTICE, '\\begin{table}[ht]', '\\centering',
             '\\caption{{{}}}'.format(caption.replace('_', '\\_')),
             '\\begin{tabular}{' + 'l' * 2 + 'c' * (len(cols) - 2) + '}', '\\hline',
             ' & '.join(cols) + ' \\\\', '\\hline']
    for i, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        if bold_rows and i in bold_rows:
            cells = ['\\textbf{{{}}}'.format(c) for c in cells]
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    with open(os.path.join(TABLES, name + '.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print('wrote', csv_path)


def main():
    with open(os.path.join(REPO_ROOT, 'results', 'paper_results', 'reported_results.json')) as f:
        r = json.load(f)
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    t1 = pd.DataFrame(r['table1_data_acquisition']['rows']).rename(columns={
        'line': 'Cultivation Line', 'variety': 'Tomato Plant Variety', 'cultivation_period': 'Cultivation Period',
        'weeks': 'Number of Weeks', 'plants': 'Number of Plants'})
    write(t1, 'table1_data_acquisition', 'Table 1: Data acquisition details')

    t3 = pd.DataFrame(r['table3_class_distribution']['rows']).rename(columns={'line': 'Dataset (Line)'})
    write(t3, 'table3_class_distribution', 'Table 3: Distribution of image samples across cultivation lines')

    t4 = pd.DataFrame(r['table4_split_sizes']['rows'])
    t4['line'] = 'Line ' + t4['line'].astype(str)
    write(t4.rename(columns={'line': 'Dataset'}), 'table4_split_sizes',
          'Table 4: Training and validation set sizes (random 80:20 split)')

    t6r = r['table6_performance_random_split']
    t6 = pd.DataFrame(t6r['rows'])
    bold = [i for i, row in t6.iterrows() if t6r['best_per_line'][str(row['line'])] == row['architecture']]
    t6 = t6.drop(columns=['line']).rename(columns={'dataset': 'Dataset', 'architecture': 'Architecture'})
    write(t6, 'table6_main_results', 'Table 6: Performance metrics across datasets (random 80:20 split)', bold)

    t7 = pd.DataFrame(r['table7_classwise_best_models']['rows']).drop(columns=['line'])
    t7 = t7.rename(columns={'dataset': 'Dataset', 'model': 'Model', 'class': 'Class'})
    write(t7, 'table7_classwise_results', 'Table 7: Class-wise performance of the best model per line')

    t8 = pd.DataFrame(r['table8_plant_based_partitioning']['rows'])
    t8['line'] = 'Line ' + t8['line'].astype(str)
    write(t8.rename(columns={'line': 'Dataset', 'architecture': 'Architecture'}), 'table8_plant_based_results',
          'Table 8: Model performance using plant-based partitioning (70:20:10 of plants)')

    # Figure 5 confusion matrices: one long-format CSV + a redrawn figure
    cm = r['figure5_confusion_matrices_row_normalized']
    long_rows, panels = [], []
    for letter, line in zip('ABCD', ['line1', 'line2', 'line3', 'line4']):
        m, names = cm[line]['matrix'], cm[line]['classes']
        for i, t in enumerate(names):
            for j, p in enumerate(names):
                long_rows.append({'line': line.replace('line', 'Line '), 'model': cm[line]['model'],
                                  'true_class': t, 'predicted_class': p, 'value': m[i][j]})
        panels.append(((letter, 'Line {} - {}'.format(line[-1], cm[line]['model'])), m, names))
    write(pd.DataFrame(long_rows), 'figure5_confusion_matrices', 'Figure 5: Row-normalised confusion matrices')

    from plant_health.plotting import plot_confusion_grid
    out = os.path.join(FIGURES, 'fig5_confusion_matrices_reported.png')
    plot_confusion_grid(panels, out, suptitle='Redrawn from the values printed in Figure 5 of the paper. ' + NOTICE)
    print('wrote', out)


if __name__ == '__main__':
    main()
