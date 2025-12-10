import json
import random
from collections import defaultdict

def find_level0_ancestor(cid, parents_dict):
    curr = cid
    while True:
        parents = parents_dict.get(curr, [])
        if not parents:
            return curr
        curr = parents[0]

# Load hierarchy
with open('Amazon_products/class_hierarchy.txt', 'r') as f:
    parents_dict = defaultdict(list)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            parent, child = map(int, parts)
            parents_dict[child].append(parent)

# Load class names
with open('Amazon_products/classes.txt', 'r') as f:
    id2class = {}
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            id2class[int(parts[0])] = parts[1]

# Load train corpus
with open('Amazon_products/train/train_corpus.txt', 'r') as f:
    train_corpus = {}
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            train_corpus[int(parts[0])] = parts[1]

# Load pseudo labels
with open('checkpoints/pseudo_labels.json', 'r') as f:
    pseudo_labels = json.load(f)

print("="*120)
print("PSEUDO-LABEL EVALUATION - Manual Inspection")
print("="*120)
print()

# Filter accepted pseudo-labels
accepted = {k: v for k, v in pseudo_labels.items() if v['accepted']}
print(f"Total unlabeled documents: {len(pseudo_labels):,}")
print(f"Accepted pseudo-labels: {len(accepted):,} ({len(accepted)/len(pseudo_labels)*100:.1f}%)")
print()

# Sample 30 documents for manual inspection
sample_size = min(30, len(accepted))
sampled_doc_ids = random.sample(list(accepted.keys()), sample_size)

print("="*120)
print(f"MANUAL INSPECTION - Random Sample of {sample_size} Pseudo-Labeled Documents")
print("="*120)
print()

perfect = 0
good = 0
questionable = 0
wrong = 0

annotations = []

for doc_id_str in sampled_doc_ids:
    doc_id = int(doc_id_str)
    doc_text = train_corpus.get(doc_id, "NOT FOUND")
    
    pred_data = accepted[doc_id_str]
    pred_classes = pred_data['selected_classes']
    confidences = pred_data['confidences']
    
    pred_names = [id2class.get(c, f'?{c}') for c in pred_classes]
    
    # Get Level 0 information
    level0_classes = set()
    for c in pred_classes:
        lv0 = find_level0_ancestor(c, parents_dict)
        level0_classes.add(lv0)
    
    level0_names = [id2class.get(lv0, f'?{lv0}') for lv0 in level0_classes]
    
    print(f"📄 Doc {doc_id}: {doc_text[:150]}...")
    print(f"   Predicted: {pred_names}")
    print(f"   Confidence: {[f'{c:.3f}' for c in confidences]}")
    print(f"   Level 0: {level0_names} (count={len(level0_classes)})")
    print(f"   Your assessment (✅ Perfect / 🟢 Good / 🟡 Questionable / ❌ Wrong):")
    print()
    
    # Store for later annotation
    annotations.append({
        'doc_id': doc_id,
        'text': doc_text,
        'predicted': pred_names,
        'confidences': confidences,
        'level0': level0_names
    })

print("="*120)
print("STATISTICS (from full dataset)")
print("="*120)

# Analyze confidence distribution
all_confidences = []
for v in accepted.values():
    all_confidences.extend(v['confidences'])

import numpy as np
print(f"\nConfidence Statistics:")
print(f"  Mean: {np.mean(all_confidences):.3f}")
print(f"  Median: {np.median(all_confidences):.3f}")
print(f"  Min: {np.min(all_confidences):.3f}")
print(f"  Max: {np.max(all_confidences):.3f}")

# Analyze number of classes per document
num_classes_dist = defaultdict(int)
for v in accepted.values():
    num_classes_dist[v['num_classes']] += 1

print(f"\nNumber of Classes per Document:")
for n in sorted(num_classes_dist.keys()):
    print(f"  {n} classes: {num_classes_dist[n]:,} docs ({num_classes_dist[n]/len(accepted)*100:.1f}%)")

# Analyze Level 0 distribution
level0_dist = defaultdict(int)
for doc_id_str, v in accepted.items():
    pred_classes = v['selected_classes']
    level0_classes = set()
    for c in pred_classes:
        lv0 = find_level0_ancestor(c, parents_dict)
        level0_classes.add(lv0)
    level0_dist[len(level0_classes)] += 1

print(f"\nLevel 0 Distribution:")
for n in sorted(level0_dist.keys()):
    print(f"  {n} Level 0: {level0_dist[n]:,} docs ({level0_dist[n]/len(accepted)*100:.1f}%)")

# Most common classes
class_freq = defaultdict(int)
for v in accepted.values():
    for c in v['selected_classes']:
        class_freq[c] += 1

print(f"\nTop 20 Most Frequent Classes:")
top_classes = sorted(class_freq.items(), key=lambda x: x[1], reverse=True)[:20]
for c, freq in top_classes:
    lv0 = find_level0_ancestor(c, parents_dict)
    lv0_name = id2class.get(lv0, f'?{lv0}')
    print(f"  {id2class.get(c, f'?{c}'):30s} (Level 0: {lv0_name:20s}): {freq:4d} times ({freq/len(accepted)*100:.1f}%)")

print()
print("="*120)
print(f"Sampled {sample_size} documents for manual inspection above")
print("="*120)
