import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_doc_candidates(file_path):
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # Collect all scores
    all_scores = []
    top1_scores = []
    top3_scores = []
    
    for doc_id, candidates in data.items():
        # candidates is {class_id: score}
        scores = list(candidates.values())
        all_scores.extend(scores)
        
        if scores:
            scores.sort(reverse=True)
            top1_scores.append(scores[0])
            if len(scores) >= 3:
                top3_scores.append(scores[2])
            else:
                top3_scores.append(scores[-1])

    print("\n=== Doc Candidates Score Distribution ===")
    print(f"Total Scores: {len(all_scores)}")
    print(f"Min Score: {min(all_scores):.4f}")
    print(f"Max Score: {max(all_scores):.4f}")
    print(f"Mean Score: {np.mean(all_scores):.4f}")
    print(f"Median Score: {np.median(all_scores):.4f}")
    
    print("\n=== Top-1 Score Stats ===")
    print(f"Mean Top-1: {np.mean(top1_scores):.4f}")
    print(f"Median Top-1: {np.median(top1_scores):.4f}")

    print("\n=== Top-3 Score Stats ===")
    print(f"Mean Top-3: {np.mean(top3_scores):.4f}")
    print(f"Median Top-3: {np.median(top3_scores):.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(all_scores, bins=50, kde=True)
    plt.title('Distribution of All Candidate Scores')
    plt.xlabel('Score')
    
    plt.subplot(1, 2, 2)
    sns.histplot(top1_scores, bins=50, kde=True, color='green', label='Top-1')
    sns.histplot(top3_scores, bins=50, kde=True, color='orange', label='Top-3', alpha=0.5)
    plt.title('Distribution of Top-1 vs Top-3 Scores')
    plt.xlabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('doc_candidates_analysis.png')
    print("\nSaved plot to doc_candidates_analysis.png")

if __name__ == "__main__":
    analyze_doc_candidates("checkpoints/doc_candidates.json")
