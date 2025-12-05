import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import utils

def analyze_scores():
    # 1. Load Data
    print("Loading data...")
    DATA_DIR = "Amazon_products"
    HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
    CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
    
    id2class, _ = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    num_classes = len(id2class)
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, num_classes)
    
    DOC_CANDIDATES_CACHE = "checkpoints/doc_candidates.json"
    with open(DOC_CANDIDATES_CACHE, 'r') as f:
        loaded_candidates = json.load(f)
        doc_candidates = {}
        for k, v in loaded_candidates.items():
            doc_candidates[int(k)] = {int(ck): cv for ck, cv in v.items()}
            
    print(f"Loaded candidates for {len(doc_candidates)} docs.")

    # 2. Calculate Confidence Scores (Replicating logic)
    print("Calculating Confidence Scores...")
    doc_confidences = {}
    class_conf_distribution = defaultdict(list)
    
    for doc_id, candidates in tqdm(doc_candidates.items(), desc="Calc Confidence"):
        doc_confidences[doc_id] = {}
        for c, score in candidates.items():
            parents = parents_dict.get(c, [])
            parent_scores = [candidates.get(p, 0.0) for p in parents]
            max_parent = max(parent_scores) if parent_scores else 0.0
            
            siblings = utils.get_siblings(c, parents_dict, children_dict)
            sibling_scores = [candidates.get(s, 0.0) for s in siblings]
            max_sibling = max(sibling_scores) if sibling_scores else 0.0
            
            conf = score - max(max_parent, max_sibling)
            doc_confidences[doc_id][c] = conf
            class_conf_distribution[c].append(conf)

    # 3. Calculate Median Thresholds
    class_thresholds = {}
    for c, scores in class_conf_distribution.items():
        if scores:
            class_thresholds[c] = np.median(scores)
        else:
            class_thresholds[c] = 0.0

    # 4. Analyze Score Drops for a sample of docs
    print("\nAnalyzing Score Drops...")
    sample_docs = list(doc_confidences.keys())[:5]
    
    for doc_id in sample_docs:
        candidates = doc_candidates[doc_id]
        confs = doc_confidences[doc_id]
        
        # Filter by threshold
        valid_candidates = []
        for c, conf in confs.items():
            tau = class_thresholds.get(c, 0.0)
            if conf > tau:
                valid_candidates.append((c, conf, candidates.get(c, 0.0)))
        
        # Sort by Confidence
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nDoc {doc_id}:")
        print(f"{'Rank':<5} {'ClassID':<10} {'Conf':<10} {'Raw':<10} {'Diff (Prev)':<10}")
        prev_conf = None
        for i, (c, conf, raw) in enumerate(valid_candidates[:20]):
            diff = f"{prev_conf - conf:.4f}" if prev_conf is not None else "-"
            print(f"{i+1:<5} {c:<10} {conf:.4f}     {raw:.4f}     {diff:<10}")
            prev_conf = conf

    # 5. Simulate Strategies
    print("\nSimulating Strategies on all docs...")
    
    strategies = {
        "Fixed Top-15": lambda scores: scores[:15],
        "Relative Drop > 0.1": lambda scores: relative_drop(scores, 0.1),
        "Relative Drop > 0.05": lambda scores: relative_drop(scores, 0.05),
        "Top-P 0.9 (Softmax)": lambda scores: top_p(scores, 0.9),
    }
    
    results = defaultdict(list)
    
    for doc_id, confs in tqdm(doc_confidences.items(), desc="Simulating"):
        # Filter and Sort
        valid_candidates = []
        for c, conf in confs.items():
            tau = class_thresholds.get(c, 0.0)
            if conf > tau:
                valid_candidates.append(conf)
        valid_candidates.sort(reverse=True)
        
        for name, func in strategies.items():
            selected = func(valid_candidates)
            results[name].append(len(selected))
            
    print("\nResults (Avg # Classes per Doc):")
    for name, counts in results.items():
        print(f"{name}: {np.mean(counts):.2f} (Min: {min(counts)}, Max: {max(counts)})")

def relative_drop(scores, threshold):
    if not scores: return []
    selected = [scores[0]]
    for i in range(1, len(scores)):
        if scores[i-1] - scores[i] > threshold:
            break
        selected.append(scores[i])
    return selected

def top_p(scores, p):
    if not scores: return []
    # Softmax over scores? Or just sum?
    # Confidence scores are not probs, so softmax might be needed if we treat them as logits.
    # But they are already somewhat normalized. Let's try simple sum normalization.
    # Actually, let's use softmax to be safe.
    scores_np = np.array(scores)
    exp_scores = np.exp(scores_np - np.max(scores_np))
    probs = exp_scores / np.sum(exp_scores)
    
    cumulative = 0.0
    selected = []
    for i, prob in enumerate(probs):
        cumulative += prob
        selected.append(scores[i])
        if cumulative >= p:
            break
    return selected

if __name__ == "__main__":
    analyze_scores()
