import json
import numpy as np
from collections import Counter

# Load data
with open('checkpoints/core_classes.json', 'r') as f:
    core_data = json.load(f)

with open('Amazon_products/classes.txt', 'r') as f:
    id2class = {}
    for line in f:
        cid, cname = line.strip().split('\t')
        id2class[cid] = cname

total_classes = len(id2class)
total_docs = len(core_data)

# Count class frequency
class_counts = Counter()
for doc_id, classes in core_data.items():
    for cid in classes:
        class_counts[cid] += 1

# Find unpredicted classes
predicted_classes = set(class_counts.keys())
all_classes = set(int(cid) for cid in id2class.keys())
never_predicted = all_classes - predicted_classes

print("="*100)
print("📊 CLASS PREDICTION DISTRIBUTION ANALYSIS")
print("="*100)

print(f"\n✅ COVERAGE:")
print(f"  Total classes: {total_classes}")
print(f"  Predicted at least once: {len(predicted_classes)} ({100*len(predicted_classes)/total_classes:.1f}%)")
print(f"  NEVER predicted: {len(never_predicted)} ({100*len(never_predicted)/total_classes:.1f}%)")
print(f"  Total documents: {total_docs}")

# Stats
predicted_counts = list(class_counts.values())
print(f"\n📈 FREQUENCY STATS:")
print(f"  Mean: {np.mean(predicted_counts):.1f} predictions/class")
print(f"  Median: {np.median(predicted_counts):.0f}")
print(f"  Std Dev: {np.std(predicted_counts):.1f}")
print(f"  Max: {max(predicted_counts)} (most over-predicted)")
print(f"  Min (non-zero): {min(predicted_counts)}")

# Top over-predicted
print(f"\n🔥 TOP 20 OVER-PREDICTED Classes:")
print(f"{'Rank':<5} {'Count':<7} {'%':<8} {'ID':<5} {'Name'}")
print("-"*100)
for i, (cid, count) in enumerate(class_counts.most_common(20), 1):
    pct = 100 * count / sum(predicted_counts)
    print(f"{i:<5} {count:<7} {pct:6.2f}%  {cid:<5} {id2class[str(cid)]}")

# Bottom under-predicted
print(f"\n❄️  TOP 20 UNDER-PREDICTED Classes (excluding 0):")
print(f"{'Rank':<5} {'Count':<7} {'ID':<5} {'Name'}")
print("-"*100)
least = sorted(class_counts.items(), key=lambda x: x[1])[:20]
for i, (cid, count) in enumerate(least, 1):
    print(f"{i:<5} {count:<7} {cid:<5} {id2class[str(cid)]}")

# Never predicted sample
print(f"\n🚫 NEVER PREDICTED Classes (sample 30):")
print(f"{'ID':<5} {'Name'}")
print("-"*50)
for cid in sorted(list(never_predicted))[:30]:
    print(f"{cid:<5} {id2class[str(cid)]}")

# Category analysis
print(f"\n🏷️  CATEGORY PATTERNS:")
over_predicted_names = [id2class[str(cid)] for cid, _ in class_counts.most_common(50)]
never_predicted_names = [id2class[str(cid)] for cid in sorted(list(never_predicted))[:50]]

# Common keywords in over-predicted
print("\nMost common words in OVER-PREDICTED classes:")
words_over = Counter()
for name in over_predicted_names:
    for word in name.split('_'):
        if len(word) > 3:
            words_over[word] += 1
for word, count in words_over.most_common(15):
    print(f"  {word}: {count}")

print("\nMost common words in NEVER-PREDICTED classes:")
words_never = Counter()
for name in never_predicted_names:
    for word in name.split('_'):
        if len(word) > 3:
            words_never[word] += 1
for word, count in words_never.most_common(15):
    print(f"  {word}: {count}")

# Save report
with open('class_distribution_report.txt', 'w') as f:
    f.write(f"Over-predicted: {class_counts.most_common(100)}\n\n")
    f.write(f"Never predicted: {sorted(never_predicted)}\n")
print("\n✅ Full report saved to class_distribution_report.txt")
