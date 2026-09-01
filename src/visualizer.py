import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import Config

def plot_optimal_k_metrics(k_values: list, inertias: list, silhouette_scores: list, optimal_k: int) -> None:
    """Plots both the Elbow Method (Inertia) and Silhouette Scores to justify optimal K."""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Elbow Method (Inertia / Within-Cluster Distance)
    ax1.plot(k_values, inertias , marker='o', linestyle='-', color='teal', linewidth=2)
    ax1.set_title('Elbow Method (Within-Cluster Distance)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=10)
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=10)
    
    # 2. Silhouette Score
    ax2.plot(k_values, silhouette_scores, marker='s', linestyle='-', color='purple', linewidth=2)
    optimal_score = silhouette_scores[k_values.index(optimal_k)]
    ax2.scatter([optimal_k], [optimal_score], color='red', s=120, zorder=5, label=f'Selected Optimal K = {optimal_k}')
    ax2.set_title('Silhouette Score (Separation)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=10)
    ax2.set_ylabel('Silhouette Score', fontsize=10)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.join(Config.BASE_DIR, "models"), exist_ok=True)
    plot_path = os.path.join(Config.BASE_DIR, "models", "optimal_k_evaluation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined Elbow & Silhouette plot saved to {plot_path}")