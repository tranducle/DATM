import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(r".")
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_learning_curve(tsv_path, metric_name, title, filename, color):
    df = pd.read_csv(tsv_path, sep='\t')
    # Filter out n/a or errors if needed, but we plot all 'experiment' iterations
    df_valid = df[df['metric_value'].notna()].copy()
    
    # Track the best metric seen so far
    best_so_far = []
    current_best = 0.0
    for val in df_valid['metric_value']:
        if val > current_best:
            current_best = val
        best_so_far.append(current_best)
        
    df_valid['best_metric'] = best_so_far
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot all attempts as scatter
    ax.scatter(df_valid['experiment'], df_valid['metric_value'], color='gray', alpha=0.5, s=20, label='Exploration')
    
    # Plot best so far line
    ax.plot(df_valid['experiment'], df_valid['best_metric'], color=color, linewidth=2.5, label=f'Optimal {metric_name}')
    
    # Highlight points where status == 'keep' (model improved)
    keeps = df_valid[df_valid['status'] == 'keep']
    ax.scatter(keeps['experiment'], keeps['metric_value'], color=color, s=50, edgecolor='black', zorder=5, label='Improvement Phase')
    
    ax.set_xlabel('Autonomous Optimisation Iteration')
    ax.set_ylabel(metric_name)
    ax.set_title(title, pad=15)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf", dpi=800, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=800, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    b_path = BASE_DIR / "results_b2.tsv"
    d_path = BASE_DIR / "results_d.tsv"
    
    if b_path.exists():
        plot_learning_curve(b_path, "AUC", "Phase B: Intent Auditing Optimization (TruthfulQA)", "fig_b_learning_curve", '#2196F3')
    else:
        print(f"Not found: {b_path}")
        
    if d_path.exists():
        plot_learning_curve(d_path, "F1-Score", "Phase D: Adversarial Tolerance Simulation", "fig_d_learning_curve", '#4CAF50')
    else:
        print(f"Not found: {d_path}")
