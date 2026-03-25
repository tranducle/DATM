#!/usr/bin/env python3
"""Generate learning curve figures from autonomous experiment TSV logs."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
DPI = 800
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

# Phase B learning curve
df_b = pd.read_csv(Path(__file__).parent / "results_b2.tsv", sep='\t')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(df_b)+1), df_b['metric_value'], '#2196F3', lw=2, marker='o', markersize=3)
best_b = df_b['metric_value'].max()
ax.axhline(best_b, color='red', ls='--', lw=1, alpha=0.7, label='Best AUC = %.4f' % best_b)
ax.set(xlabel='Autonomous Iteration', ylabel='AUC',
       title='Experiment 2: Intent Auditing Optimization (TruthfulQA)')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG / 'fig_exp2_learning.pdf', dpi=DPI, bbox_inches='tight')
fig.savefig(FIG / 'fig_exp2_learning.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('Saved fig_exp2_learning')

# Phase D learning curve
df_d = pd.read_csv(Path(__file__).parent / "results_d.tsv", sep='\t')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(df_d)+1), df_d['metric_value'], '#FF5722', lw=2, marker='o', markersize=3)
best_d = df_d['metric_value'].max()
ax.axhline(best_d, color='red', ls='--', lw=1, alpha=0.7, label='Best F1 = %.4f' % best_d)
ax.set(xlabel='Autonomous Iteration', ylabel='Mean F1-Score',
       title='Experiment 4: Adversarial Robustness Optimization (Monte Carlo)')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG / 'fig_exp4_learning.pdf', dpi=DPI, bbox_inches='tight')
fig.savefig(FIG / 'fig_exp4_learning.png', dpi=DPI, bbox_inches='tight')
plt.close()
print('Saved fig_exp4_learning')
print('DONE')
