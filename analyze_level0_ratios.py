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

def analyze_level0_ratios():
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

    # 2. Calculate Confidence Scores & Level 0 Scores
    ratios = []
    
    # Helper to find Level 0
    def get_level0_ancestor(class_id):
        current = class_id
        while True:
            parents = parents_dict.get(current, [])
            if not parents:
                return current
            current = parents[0]

    for doc_id, candidates in tqdm(doc_candidates.items(), desc="Analyzing Ratios"):
        # Calculate Confidence
        confs = {}
        for c, score in candidates.items():
            parents = parents_dict.get(c, [])
            parent_scores = [candidates.get(p, 0.0) for p in parents]
            max_parent = max(parent_scores) if parent_scores else 0.0
            
            siblings = utils.get_siblings(c, parents_dict, children_dict)
            sibling_scores = [candidates.get(s, 0.0) for s in siblings]
            max_sibling = max(sibling_scores) if sibling_scores else 0.0
            
            conf = score - max(max_parent, max_sibling)
            confs[c] = conf
            
        # Filter Candidates (Relative Drop Step)
        sorted_candidates = sorted(confs.items(), key=lambda x: x[1], reverse=True)
        selected_candidates = []
        if sorted_candidates:
            selected_candidates.append(sorted_candidates[0])
            for i in range(1, len(sorted_candidates)):
                # Relative Drop > 0.1
                if sorted_candidates[i-1][1] - sorted_candidates[i][1] > 0.1:
                    break
                # Max K = 10
                if len(selected_candidates) >= 10:
                    break
                selected_candidates.append(sorted_candidates[i])
        
        if not selected_candidates:
            continue
            
        # Calculate Level 0 Scores (Sum)
        level0_scores = defaultdict(float)
        for c, conf in selected_candidates:
            l0 = get_level0_ancestor(c)
            level0_scores[l0] += conf
            
        sorted_l0 = sorted(level0_scores.values(), reverse=True)
        
        if len(sorted_l0) >= 2:
            # Ratio: 2nd / 1st
            if sorted_l0[0] > 0:
                ratio = sorted_l0[1] / sorted_l0[0]
            else:
                ratio = 0.0
            ratios.append(ratio)
        else:
            # Only 1 Level 0 found -> Ratio is effectively 0 (or undefined, but effectively 2nd is 0)
            ratios.append(0.0)

    # 3. Analyze Distribution
    ratios = np.array(ratios)
    print("\n=== Level 0 Ratio Analysis (2nd / 1st) ===")
    print(f"Total Docs: {len(ratios)}")
    print(f"Docs with only 1 Level 0 (Ratio 0.0): {np.sum(ratios == 0.0)} ({np.mean(ratios == 0.0)*100:.1f}%)")
    
    non_zero_ratios = ratios[ratios > 0]
    print(f"Docs with >1 Level 0: {len(non_zero_ratios)}")
    if len(non_zero_ratios) > 0:
        print(f"Mean Ratio: {np.mean(non_zero_ratios):.4f}")
        print(f"Median Ratio: {np.median(non_zero_ratios):.4f}")
        print(f"25th Percentile: {np.percentile(non_zero_ratios, 25):.4f}")
        print(f"75th Percentile: {np.percentile(non_zero_ratios, 75):.4f}")
        
        # Percentiles for potential thresholds
        print("\nPotential Thresholds (How many kept?):")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            kept_count = np.sum(ratios >= thresh)
            print(f"Threshold {thresh}: Keep {kept_count} docs as Multi-L0 ({kept_count/len(ratios)*100:.1f}% of total)")

if __name__ == "__main__":
    analyze_level0_ratios()
