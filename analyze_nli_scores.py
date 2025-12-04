
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_nli_scores(candidates_path, core_classes_path):
    print(f"Loading candidates from {candidates_path}...")
    try:
        with open(candidates_path, 'r') as f:
            candidates = json.load(f)
    except Exception as e:
        print(f"Error loading candidates: {e}")
        return

    print(f"Loading core classes from {core_classes_path}...")
    try:
        with open(core_classes_path, 'r') as f:
            core_classes = json.load(f)
    except Exception as e:
        print(f"Error loading core classes: {e}")
        return

    # 1. Collect all scores from candidates
    all_scores = []
    for doc_id, cands in candidates.items():
        for cid, score in cands.items():
            all_scores.append(score)
            
    all_scores = np.array(all_scores)
    
    # 2. Collect scores of selected core classes
    selected_scores = []
    for doc_id, cores in core_classes.items():
        # core_classes.json is list of class IDs
        # We need to look up their scores in candidates
        cands = candidates.get(doc_id, {})
        # Note: doc_id in json keys are strings
        if isinstance(cands, dict):
             # Try to match keys (int vs str)
             pass
        
        # Re-load logic to handle str keys
        cands = candidates.get(str(doc_id), {})
        if not cands:
             cands = candidates.get(doc_id, {})
             
        for cid in cores:
            # cid might be int, keys in json are str
            score = cands.get(str(cid))
            if score is None:
                score = cands.get(cid)
            
            if score is not None:
                selected_scores.append(score)
                
    selected_scores = np.array(selected_scores)

    print("\n" + "="*40)
    print("📊 NLI Score Analysis")
    print("="*40)
    
    print(f"Total Candidates: {len(all_scores)}")
    print(f"Total Selected Core Classes: {len(selected_scores)}")
    
    print("\n[All Candidates Stats]")
    print(f"Min: {all_scores.min():.4f}")
    print(f"Max: {all_scores.max():.4f}")
    print(f"Mean: {all_scores.mean():.4f}")
    print(f"Median: {np.median(all_scores):.4f}")
    
    # Threshold Analysis
    thresholds = [0.1, 0.2, 0.33, 0.5, 0.7, 0.9]
    print("\n[Threshold Analysis - Candidates below threshold]")
    for t in thresholds:
        count = np.sum(all_scores < t)
        ratio = count / len(all_scores) * 100
        print(f" < {t}: {count} ({ratio:.2f}%)")
        
    print("\n[Selected Core Classes Stats]")
    if len(selected_scores) > 0:
        print(f"Min: {selected_scores.min():.4f}")
        print(f"Max: {selected_scores.max():.4f}")
        print(f"Mean: {selected_scores.mean():.4f}")
        print(f"Median: {np.median(selected_scores):.4f}")
        
        print("\n[Selected Core Classes below 0.33]")
        low_conf_cores = np.sum(selected_scores < 0.33)
        print(f"Count: {low_conf_cores} ({low_conf_cores/len(selected_scores)*100:.2f}%)")
    else:
        print("No selected scores found (check ID matching).")

    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(all_scores, bins=50, kde=True, color='skyblue')
    plt.axvline(0.33, color='r', linestyle='--', label='Threshold 0.33')
    plt.title('Distribution of All Candidate Scores')
    plt.xlabel('NLI Entailment Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if len(selected_scores) > 0:
        sns.histplot(selected_scores, bins=50, kde=True, color='orange')
        plt.axvline(0.33, color='r', linestyle='--', label='Threshold 0.33')
        plt.title('Distribution of Selected Core Class Scores')
        plt.xlabel('NLI Entailment Score')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('nli_score_analysis.png')
    print("\nSaved plot to nli_score_analysis.png")

if __name__ == "__main__":
    analyze_nli_scores(
        'checkpoints/doc_candidates.json',
        'checkpoints/core_classes.json'
    )
