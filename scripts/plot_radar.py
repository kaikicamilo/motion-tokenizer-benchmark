"""Radar chart of quantizer profiles on SnapMoGen (paper figure).

Six axes, each min-max normalized over the non-collapsed methods; outer = better,
except codebook usage, which is a diagnostic rather than a quality objective.

    python scripts/plot_radar.py --metrics metrics_snapmogen.csv
"""
import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, GRID = '#0b0b0b', '#d3d2ce'
SHOWN = [('Res-FSQ', ORANGE, 'Res-FSQ'), ('MS-RVQ [8,4,2,1]', YELLOW, 'MS-RVQ [8,4,2,1]'),
         ('Shared RVQ x6', BLUE, 'Shared RVQ $\\times$6'), ('RVQ x6', AQUA, 'RVQ $\\times$6')]
AXES = [('Fidelity (FID)', 'FID', False), ('Semantics\n(R-Prec@3)', 'R_Prec_Top3', True),
        ('Geometry\n(MPJPE)', 'MPJPE_cm', False), ('Codebook usage\n(perplexity)', 'perplexity', True),
        ('Token\nefficiency', 'tokens_step', False), ('Compactness\n(fewer params)', 'size_M', False)]

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'text.color': INK, 'savefig.bbox': 'tight'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True, help='CSV from collect_metrics.py (+ MPJPE_cm column)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'spg_fig1_radar'))
    a = ap.parse_args()

    rows = {r['method']: r for r in csv.DictReader(open(a.metrics)) if r['method'] != 'LFQ'}
    norm = {}
    for _, key, higher in AXES:
        vals = [float(r[key]) for r in rows.values()]; lo, hi = min(vals), max(vals)
        norm[key] = (lambda v, lo=lo, hi=hi, higher=higher: (v - lo) / (hi - lo) if higher else (hi - v) / (hi - lo))
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    closed = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(6.4, 5.8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for name, color, _ in SHOWN:
        v = [norm[k](float(rows[name][k])) for _, k, _ in AXES]; v += v[:1]
        ax.plot(closed, v, color=color, lw=2); ax.fill(closed, v, color=color, alpha=0.12)
        ax.scatter(ang, v[:-1], s=22, color=color, edgecolor='white', lw=0.8, zorder=4)
    labels = ax.set_xticklabels([]); ax.set_xticks(ang)
    labels = ax.set_xticklabels([l for l, _, _ in AXES], fontsize=12.5, color=INK)
    ax.tick_params(axis='x', pad=12)
    for l in labels:
        l.set_zorder(30); l.set_path_effects([withStroke(linewidth=3, foreground='white')])
    ax.set_ylim(0, 1.18); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([])
    ax.grid(color=GRID, lw=0.7); ax.spines['polar'].set_visible(False)

    handles = [Line2D([0], [0], color=c, lw=3.4, label=lbl) for _, c, lbl in SHOWN]
    kw = dict(loc='lower center', frameon=False, fontsize=16, labelcolor=INK, handletextpad=0.6, columnspacing=1.6)
    leg1 = ax.legend(handles=handles[:2], bbox_to_anchor=(0.5, -0.26), ncol=2, **kw); ax.add_artist(leg1)
    leg2 = ax.legend(handles=[handles[3], handles[2]], bbox_to_anchor=(0.5, -0.35), ncol=2, **kw)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=300, bbox_extra_artists=(leg1, leg2))
    print('saved', a.out + '.{pdf,png}')


if __name__ == '__main__':
    main()
