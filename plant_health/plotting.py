"""Plotting helpers for the paper's figures.

Figure 5  confusion matrices (row-normalised)      -> plot_confusion_matrix / plot_confusion_grid
Figure 6  t-SNE of penultimate features            -> plot_tsne
Figure 7-8 Grad-CAM overlays                       -> plot_gradcam_grid
Figure 9  qualitative predictions with confidence  -> plot_prediction_grid
Figure 10 predicted vs expert state over time      -> plot_prediction_vs_expert
Figure 11 expert state trajectories (all plants)   -> plot_expert_trajectories
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LEVEL_COLORS = {0: '#000000', 1: '#b2182b', 2: '#ef8a62', 3: '#e0c341', 4: '#67a9cf', 5: '#1a9850'}


def _annotate_value(v):
    return '{:.2g}'.format(v) if v < 0.1 else '{:.2f}'.format(v).rstrip('0').rstrip('.')


def plot_confusion_matrix(matrix, class_names, path=None, ax=None, title='Confusion Matrix'):
    matrix = np.asarray(matrix, dtype=float)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1e-9))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, _annotate_value(v), ha='center', va='center', fontsize=9,
                    color='white' if v > 0.5 * matrix.max() else '#222222')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, rotation=90, va='center')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if own:
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=200)
        plt.close()


def plot_confusion_grid(panels, path, suptitle=None):
    """panels: list of (label, matrix, class_names); 2 x 2 grid like Figure 5."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, (label, m, names) in zip(axes.ravel(), panels):
        plot_confusion_matrix(m, names, ax=ax, title='({}) {}'.format(*label) if isinstance(label, tuple) else label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_tsne(embedding, labels, class_names, path, title='t-SNE'):
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, name in enumerate(class_names):
        m = labels == k
        level = int(name.split()[-1]) if name.split()[-1].isdigit() else k
        ax.scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.7, label=name,
                   color=LEVEL_COLORS.get(level))
    ax.legend(markerscale=3, frameon=False)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_gradcam_grid(images, cams, captions, path, ncols=4):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, img, cam, cap in zip(axes.ravel(), images, cams, captions):
        ax.imshow(img)
        ax.imshow(cam, cmap='jet', alpha=0.4)
        ax.set_title(cap, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_grid(images, captions, path, ncols=4):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, img, cap in zip(axes.ravel(), images, captions):
        ax.imshow(img)
        ax.set_title(cap, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_vs_expert(traj, path, title=''):
    """traj: DataFrame with week, expert_level, predicted_level for one plant (Figure 10)."""
    traj = traj.sort_values('week')
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(traj['week'], traj['expert_level'], 'r--', lw=1.5, label='Expert label (ground truth)')
    ax.plot(traj['week'], traj['predicted_level'], 'o-', color='#2166ac', ms=4, lw=1, label='Model prediction')
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel('Week')
    ax.set_ylabel('Health status')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_expert_trajectories(frame, path, title='', window=3):
    """frame: per (plant_id, week) expert level incl. 0 = dead (Figure 11).

    Points show every plant-week; the dashed line is the moving average of the mean
    state per week.
    """
    from plant_health.analysis.temporal import moving_average
    fig, ax = plt.subplots(figsize=(9, 3.5))
    plants = sorted(frame['plant_id'].unique(), key=lambda s: (0, int(s)) if str(s).isdigit() else (1, str(s)))
    offsets = np.linspace(-0.3, 0.3, max(1, len(plants)))
    for off, p in zip(offsets, plants):
        g = frame[frame['plant_id'] == p].sort_values('week')
        ax.scatter(g['week'] + off, g['health_level'], s=6, c=[LEVEL_COLORS[int(v)] for v in g['health_level']])
    weekly = frame.groupby('week')['health_level'].mean()
    ax.plot(weekly.index, moving_average(weekly.values, window), 'b--', lw=1.5, label='moving average')
    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel('Week')
    ax.set_ylabel('Health state (0 = dead)')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
