"""Reading the expert annotation file and applying the class filtering of Sec. 3.1.1.

Expected columns of ``data/annotations.csv`` (one row per image; see data/README.md):

    image_path    path relative to ``base.image_root``
    line          cultivation line, 1-4 (Table 1)
    plant_id      plant number within the line (from the QR tag, Sec. 2.1)
    week          week index within the cultivation period (1-25 for Lines 2-3, 1-18 for Lines 1 and 4)
    view          left | right | top (three viewpoints per plant per visit, Sec. 2.1)
    health_level  expert rating 1-5 (Table 2); 0 marks a dead plant (used only for Figure 11)

The column layout is an [ASSUMPTION] of this repository: the paper describes the
information that was recorded (plant number, cultivation period, weekly visits,
three viewpoints, expert level) but not the file format.
"""
import pandas as pd

REQUIRED_COLUMNS = ['image_path', 'line', 'plant_id', 'week', 'view', 'health_level']
VIEWS = ('left', 'right', 'top')
DEAD_LEVEL = 0


def load_annotations(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError('annotations file {} is missing columns: {}'.format(path, missing))
    df['line'] = df['line'].astype(int)
    df['week'] = df['week'].astype(int)
    df['health_level'] = df['health_level'].astype(int)
    df['plant_id'] = df['plant_id'].astype(str)
    df['view'] = df['view'].astype(str).str.lower()
    bad = set(df['health_level'].unique()) - {DEAD_LEVEL, 1, 2, 3, 4, 5}
    if bad:
        raise ValueError('health_level must be 0-5, found {}'.format(sorted(bad)))
    return df


def select_line(df, line):
    return df[df['line'] == int(line)].reset_index(drop=True)


def classification_subset(df):
    """Images usable for classification: expert levels 1-5 (dead plants, level 0, are dropped)."""
    return df[df['health_level'] != DEAD_LEVEL].reset_index(drop=True)


def class_distribution(df, levels=(1, 2, 3, 4, 5)):
    """Per-line count of images per health level (reproduces the layout of Table 3)."""
    df = classification_subset(df)
    table = df.pivot_table(index='line', columns='health_level', values='image_path',
                           aggfunc='count', fill_value=0)
    table = table.reindex(columns=list(levels), fill_value=0)
    table['Total'] = table.sum(axis=1)
    table.loc['Total'] = table.sum(axis=0)
    return table


def retained_levels(df_line, levels, min_samples, exclude_at_threshold=True):
    """Health levels kept for one line after removing under-represented classes.

    Sec. 3.1.1: "classes with fewer than 30 samples (representing fewer than 10 plants)
    were excluded". Figure 5C shows that Class 1 of Line 3 (30 samples in Table 3) was
    also removed, so by default a class is dropped when count <= min_samples.
    """
    counts = df_line['health_level'].value_counts()
    kept = []
    for level in levels:
        n = int(counts.get(level, 0))
        drop = n <= min_samples if exclude_at_threshold else n < min_samples
        if not drop:
            kept.append(int(level))
    return kept
