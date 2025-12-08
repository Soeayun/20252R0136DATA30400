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
with open('ground_truth_100samples.json', 'r') as f:
    ground_truth = json.load(f)

# Load current core classes
with open('checkpoints/core_classes.json', 'r') as f:
    core_classes = json.load(f)

print("="*120)
print("GROUND TRUTH EVALUATION - 100 SAMPLES")
print("="*120)
print()

total = 0
perfect_match = 0
partial_match = 0
wrong = 0
excessive = 0

errors = []

for doc_id_str, gt_data in ground_truth.items():
    doc_id = int(doc_id_str)
    total += 1
    
    # Get predicted classes
    predicted_ids = core_classes.get(doc_id_str, [])
    predicted_names = [id2class.get(c, f'?{c}') for c in predicted_ids]
    
    # Ground truth
    gt_names = gt_data['ground_truth']
    text = gt_data['text']
    
    # Calculate metrics
    gt_set = set(gt_names)
    pred_set = set(predicted_names)
    
    correct = gt_set & pred_set
    missing = gt_set - pred_set
    extra = pred_set - gt_set
    
    # Categorize
    status = ""
    if pred_set == gt_set:
        status = "✅ PERFECT"
        perfect_match += 1
    elif len(correct) > 0:
        if len(extra) > len(missing) and len(extra) >= 3:
            status = "🟡 EXCESSIVE"
            excessive += 1
        else:
            status = "🟡 PARTIAL"
            partial_match += 1
    else:
        status = "❌ WRONG"
        wrong += 1
    
    # Log errors
    if status != "✅ PERFECT":
        errors.append({
            'doc_id': doc_id,
            'text': text,
            'status': status,
            'gt': gt_names,
            'pred': predicted_names,
            'correct': list(correct),
            'missing': list(missing),
            'extra': list(extra)
        })

# Print summary
print(f"{'='*120}")
print("SUMMARY")
print(f"{'='*120}")
print(f"Total samples: {total}")
print(f"✅ Perfect match:  {perfect_match:2d} ({perfect_match/total*100:.1f}%)")
print(f"🟡 Partial match:  {partial_match:2d} ({partial_match/total*100:.1f}%)")
print(f"🟡 Excessive:      {excessive:2d} ({excessive/total*100:.1f}%)")
print(f"❌ Wrong:          {wrong:2d} ({wrong/total*100:.1f}%)")
print()

# Precision/Recall
total_gt = sum(len(gt_data['ground_truth']) for gt_data in ground_truth.values())
total_pred = sum(len(core_classes.get(doc_id, [])) for doc_id in ground_truth.keys())
total_correct = sum(len(set(ground_truth[doc_id]['ground_truth']) & 
                          set([id2class.get(c, '') for c in core_classes.get(doc_id, [])]))
                    for doc_id in ground_truth.keys())

precision = total_correct / total_pred if total_pred > 0 else 0
recall = total_correct / total_gt if total_gt > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nAvg classes per doc (GT):   {total_gt/total:.1f}")
print(f"Avg classes per doc (Pred): {total_pred/total:.1f}")
print()

# Print errors
print(f"{'='*120}")
print("DETAILED ERRORS")
print(f"{'='*120}")
print()

for err in errors[:20]:  # Show first 20
    print(f"{err['status']} Doc {err['doc_id']}: {err['text']}")
    print(f"  GT:      {err['gt']}")
    print(f"  Pred:    {err['pred']}")
    if err['correct']:
        print(f"  ✓ Correct: {err['correct']}")
    if err['missing']:
        print(f"  ✗ Missing: {err['missing']}")
    if err['extra']:
        print(f"  + Extra:   {err['extra']}")
    print()

print(f"{'='*120}")
print(f"Showing {min(20, len(errors))}/{len(errors)} errors")
print(f"{'='*120}")
