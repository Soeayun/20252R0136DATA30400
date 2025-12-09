import json
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

# Load ground truth
with open('ground_truth_150samples.json', 'r') as f:
    ground_truth = json.load(f)

# Load BOTH versions
with open('checkpoints/core_classes.json', 'r') as f:
    core_classes_original = json.load(f)

with open('checkpoints/core_classes_llm_refined.json', 'r') as f:
    core_classes_llm = json.load(f)

def evaluate_version(core_classes, version_name):
    """Evaluate one version"""
    total = 0
    perfect_match = 0
    partial_match = 0
    wrong = 0
    excessive = 0
    
    total_gt = 0
    total_pred = 0
    total_correct = 0
    
    for doc_id_str, gt_data in ground_truth.items():
        total += 1
        
        predicted_ids = core_classes.get(doc_id_str, [])
        predicted_names = [id2class.get(c, f'?{c}') for c in predicted_ids]
        
        gt_names = gt_data['ground_truth']
        
        gt_set = set(gt_names)
        pred_set = set(predicted_names)
        
        correct = gt_set & pred_set
        missing = gt_set - pred_set
        extra = pred_set - gt_set
        
        total_gt += len(gt_set)
        total_pred += len(pred_set)
        total_correct += len(correct)
        
        # Categorize
        if pred_set == gt_set:
            perfect_match += 1
        elif len(correct) > 0:
            if len(extra) > len(missing) and len(extra) >= 3:
                excessive += 1
            else:
                partial_match += 1
        else:
            wrong += 1
    
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'version': version_name,
        'total': total,
        'perfect': perfect_match,
        'partial': partial_match,
        'excessive': excessive,
        'wrong': wrong,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_classes': total_pred / total
    }

# Evaluate both
results_original = evaluate_version(core_classes_original, 'Original (No LLM)')
results_llm = evaluate_version(core_classes_llm, 'LLM Refined (10K)')

print("="*120)
print("COMPARISON: ORIGINAL vs LLM REFINED (150 SAMPLES)")
print("="*120)
print()

# Print comparison table
print(f"{'Metric':<25} {'Original':<20} {'LLM Refined':<20} {'Difference':<20}")
print("-"*120)

metrics = [
    ('Perfect Match', 'perfect', '%'),
    ('Partial Match', 'partial', '%'),
    ('Excessive', 'excessive', '%'),
    ('Wrong', 'wrong', '%'),
    ('', '', ''),
    ('Precision', 'precision', 'score'),
    ('Recall', 'recall', 'score'),
    ('F1 Score', 'f1', 'score'),
    ('', '', ''),
    ('Avg Classes/Doc', 'avg_classes', 'num'),
]

for metric_name, key, fmt in metrics:
    if not metric_name:
        print()
        continue
    
    if fmt == '%':
        orig_val = results_original[key] / results_original['total'] * 100
        llm_val = results_llm[key] / results_llm['total'] * 100
        diff = llm_val - orig_val
        print(f"{metric_name:<25} {orig_val:>6.1f}%              {llm_val:>6.1f}%              {diff:>+6.1f}%")
    elif fmt == 'score':
        orig_val = results_original[key]
        llm_val = results_llm[key]
        diff = llm_val - orig_val
        symbol = '✅' if diff > 0 else '❌' if diff < 0 else '='
        print(f"{metric_name:<25} {orig_val:>6.3f}              {llm_val:>6.3f}              {diff:>+6.3f} {symbol}")
    elif fmt == 'num':
        orig_val = results_original[key]
        llm_val = results_llm[key]
        diff = llm_val - orig_val
        symbol = '✅' if diff < 0 else '❌' if diff > 0 else '='
        print(f"{metric_name:<25} {orig_val:>6.1f}              {llm_val:>6.1f}              {diff:>+6.1f} {symbol}")

print()
print("="*120)
print("DETAILED DIFFERENCES (Sample-by-Sample)")
print("="*120)
print()

# Find cases where they differ significantly
improvements = []
regressions = []
no_change = []

for doc_id_str, gt_data in ground_truth.items():
    orig_pred = set([id2class.get(c, f'?{c}') for c in core_classes_original.get(doc_id_str, [])])
    llm_pred = set([id2class.get(c, f'?{c}') for c in core_classes_llm.get(doc_id_str, [])])
    gt_set = set(gt_data['ground_truth'])
    
    orig_correct = len(gt_set & orig_pred)
    llm_correct = len(gt_set & llm_pred)
    
    if llm_correct > orig_correct:
        improvements.append((doc_id_str, gt_data['text'], orig_pred, llm_pred, gt_set))
    elif llm_correct < orig_correct:
        regressions.append((doc_id_str, gt_data['text'], orig_pred, llm_pred, gt_set))
    else:
        no_change.append(doc_id_str)

print(f"Improvements (LLM better): {len(improvements)}")
print(f"Regressions (LLM worse):   {len(regressions)}")
print(f"No change:                 {len(no_change)}")
print()

# Show top improvements
if improvements:
    print("="*120)
    print("TOP 10 IMPROVEMENTS (LLM Refined Better)")
    print("="*120)
    for doc_id, text, orig, llm, gt in improvements[:10]:
        print(f"\nDoc {doc_id}: {text}")
        print(f"  Ground Truth: {sorted(gt)}")
        print(f"  Original:     {sorted(orig)} ❌")
        print(f"  LLM Refined:  {sorted(llm)} ✅")

print()

# Show top regressions
if regressions:
    print("="*120)
    print("TOP 10 REGRESSIONS (LLM Refined Worse)")
    print("="*120)
    for doc_id, text, orig, llm, gt in regressions[:10]:
        print(f"\nDoc {doc_id}: {text}")
        print(f"  Ground Truth: {sorted(gt)}")
        print(f"  Original:     {sorted(orig)} ✅")
        print(f"  LLM Refined:  {sorted(llm)} ❌")

print()
print("="*120)
