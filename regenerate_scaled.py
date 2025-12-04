import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.getcwd())

def analyze_raw_score_distribution():
    DOC_CANDIDATES_CACHE = os.path.join("checkpoints", "doc_candidates.json")

    print(f"Loading Doc Candidates from {DOC_CANDIDATES_CACHE}...")
    with open(DOC_CANDIDATES_CACHE, 'r') as f:
        loaded_candidates = json.load(f)
        
    print("Collecting Raw Scores (No Scaling)...")
    all_raw_scores = []
    
    for k, v in loaded_candidates.items():
        scores = list(v.values())
        all_raw_scores.extend(scores)

    print(f"\nTotal Raw Scores: {len(all_raw_scores)}")
    print(f"Mean: {np.mean(all_raw_scores):.6f}")
    print(f"Median: {np.median(all_raw_scores):.6f}")
    print(f"Min: {np.min(all_raw_scores):.6f}")
    print(f"Max: {np.max(all_raw_scores):.6f}")

    # Calculate Percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    perc_values = np.percentile(all_raw_scores, percentiles)
    
    print("\n=== Raw Score Percentiles ===")
    for p, val in zip(percentiles, perc_values):
        print(f"{p}%: {val:.6f}")

    # Plot Score Distribution
    plt.figure(figsize=(12, 6))
    
    # Histogram
    ax = sns.histplot(all_raw_scores, bins=100, kde=True, color='blue')
    plt.title('Distribution of Raw Scores (All Candidates)')
    plt.xlabel('Raw Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add vertical lines for percentiles
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    for p, val, color in zip(percentiles, perc_values, colors):
        plt.axvline(val, color=color, linestyle='--', alpha=0.8, label=f'{p}%: {val:.6f}')

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    output_path = 'raw_score_distribution.png'
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")

if __name__ == "__main__":
    analyze_raw_score_distribution()
