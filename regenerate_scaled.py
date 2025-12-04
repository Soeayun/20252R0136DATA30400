import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.getcwd())

def analyze_score_distribution_after_scaling():
    DOC_CANDIDATES_CACHE = os.path.join("checkpoints", "doc_candidates.json")

    print(f"Loading Doc Candidates from {DOC_CANDIDATES_CACHE}...")
    with open(DOC_CANDIDATES_CACHE, 'r') as f:
        loaded_candidates = json.load(f)
        
    print("Applying Min-Max Scaling per Document...")
    all_scaled_scores = []
    
    for k, v in loaded_candidates.items():
        scores = np.array(list(v.values()))
        
        if len(scores) > 0:
            min_p = np.min(scores)
            max_p = np.max(scores)
            if max_p > min_p:
                scores = (scores - min_p) / (max_p - min_p)
            else:
                scores = np.zeros_like(scores) 
        
        all_scaled_scores.extend(scores)

    print(f"\nTotal Scaled Scores: {len(all_scaled_scores)}")
    print(f"Mean: {np.mean(all_scaled_scores):.4f}")
    print(f"Median: {np.median(all_scaled_scores):.4f}")
    print(f"Min: {np.min(all_scaled_scores):.4f}")
    print(f"Max: {np.max(all_scaled_scores):.4f}")

    # Calculate Percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    perc_values = np.percentile(all_scaled_scores, percentiles)
    
    print("\n=== Percentiles ===")
    for p, val in zip(percentiles, perc_values):
        print(f"{p}%: {val:.4f}")

    # Plot Score Distribution
    plt.figure(figsize=(12, 6))
    
    # Histogram
    ax = sns.histplot(all_scaled_scores, bins=50, kde=True, color='purple')
    plt.title('Distribution of Min-Max Scaled Scores (All Candidates)')
    plt.xlabel('Scaled Score (0.0 - 1.0)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add vertical lines for percentiles
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    for p, val, color in zip(percentiles, perc_values, colors):
        plt.axvline(val, color=color, linestyle='--', alpha=0.8, label=f'{p}%: {val:.2f}')
        # Add text annotation
        # plt.text(val, ax.get_ylim()[1]*0.9, f'{p}%', rotation=90, verticalalignment='top', color=color)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('scaled_score_distribution.png')
    print("\nSaved plot to scaled_score_distribution.png")

if __name__ == "__main__":
    analyze_score_distribution_after_scaling()
