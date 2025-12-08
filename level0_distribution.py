import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

def get_level0_mapping():
    """
    각 클래스가 어느 level 0 클래스에 속하는지 매핑
    Returns: {class_id: level0_class_id}
    """
    # Load hierarchy
    with open('Amazon_products/class_hierarchy.txt', 'r') as f:
        edges = []
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = map(int, parts)
                edges.append((parent, child))
    
    # Build parent dict
    parents_dict = defaultdict(list)
    for parent, child in edges:
        parents_dict[child].append(parent)
    
    # Load all classes
    with open('Amazon_products/classes.txt', 'r') as f:
        all_classes = set()
        id2class = {}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cid, cname = parts
                all_classes.add(int(cid))
                id2class[int(cid)] = cname
    
    # Find root nodes (level 0)
    level0_nodes = set()
    for cid in all_classes:
        if cid not in parents_dict or len(parents_dict[cid]) == 0:
            level0_nodes.add(cid)
    
    print(f"Found {len(level0_nodes)} Level 0 nodes:")
    for cid in sorted(level0_nodes):
        print(f"  Class {cid}: {id2class.get(cid, 'Unknown')}")
    
    # Map each class to its root (level 0) ancestor
    class_to_level0 = {}
    
    def find_root(cid):
        """Recursively find the root ancestor"""
        if cid in level0_nodes:
            return cid
        if cid not in parents_dict or len(parents_dict[cid]) == 0:
            return cid  # If no parents, it's a root itself
        # Follow the first parent (assuming tree structure)
        parent = parents_dict[cid][0]
        return find_root(parent)
    
    for cid in all_classes:
        class_to_level0[cid] = find_root(cid)
    
    return class_to_level0, id2class, level0_nodes

def analyze_level0_distribution():
    """
    Analyze how many distinct level 0 classes each document has in its core classes
    """
    print("="*100)
    print("Analyzing Level 0 Class Distribution in Core Classes")
    print("="*100)
    
    # Get mapping
    class_to_level0, id2class, level0_nodes = get_level0_mapping()
    
    # Load core classes
    print("\nLoading core classes...")
    with open('checkpoints/core_classes.json', 'r') as f:
        core_classes = json.load(f)
    
    # Analyze level 0 distribution
    level0_counts = []  # Number of distinct level 0 classes per document
    level0_distribution = Counter()  # Count of documents with X level 0 classes
    
    docs_with_classes = 0
    total_docs = len(core_classes)
    
    for doc_id, classes in core_classes.items():
        if not classes:
            level0_counts.append(0)
            level0_distribution[0] += 1
            continue
        
        docs_with_classes += 1
        
        # Get unique level 0 classes for this document
        level0_set = set()
        for cid in classes:
            level0_id = class_to_level0.get(int(cid), -1)
            if level0_id != -1:
                level0_set.add(level0_id)
        
        num_level0 = len(level0_set)
        level0_counts.append(num_level0)
        level0_distribution[num_level0] += 1
    
    # Statistics
    print(f"\n{'='*100}")
    print(f"Summary Statistics:")
    print(f"{'='*100}")
    print(f"Total documents: {total_docs}")
    print(f"Documents with core classes: {docs_with_classes} ({docs_with_classes/total_docs*100:.1f}%)")
    print(f"Documents without core classes: {total_docs - docs_with_classes} ({(total_docs - docs_with_classes)/total_docs*100:.1f}%)")
    
    non_zero_counts = [c for c in level0_counts if c > 0]
    if non_zero_counts:
        print(f"\nLevel 0 Class Count Statistics (excluding empty docs):")
        print(f"  Mean: {np.mean(non_zero_counts):.2f}")
        print(f"  Median: {np.median(non_zero_counts):.1f}")
        print(f"  Min: {min(non_zero_counts)}")
        print(f"  Max: {max(non_zero_counts)}")
        print(f"  Std: {np.std(non_zero_counts):.2f}")
    
    print(f"\n{'='*100}")
    print(f"Distribution of Level 0 Classes per Document:")
    print(f"{'='*100}")
    print(f"{'Num Level 0':<15} {'Count':<10} {'Percentage':<12} {'Bar':<50}")
    print(f"-"*100)
    
    for num_level0 in sorted(level0_distribution.keys()):
        count = level0_distribution[num_level0]
        pct = count / total_docs * 100
        bar = '█' * int(pct)
        print(f"{num_level0:<15} {count:<10} {pct:>5.1f}%       {bar:<50}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bar chart of distribution
    sorted_keys = sorted(level0_distribution.keys())
    counts = [level0_distribution[k] for k in sorted_keys]
    
    axes[0, 0].bar(sorted_keys, counts, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Level 0 Classes', fontsize=12)
    axes[0, 0].set_ylabel('Number of Documents', fontsize=12)
    axes[0, 0].set_title('Distribution of Level 0 Classes per Document', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_xticks(sorted_keys)
    
    # Plot 2: Pie chart (excluding 0)
    non_zero_dist = {k: v for k, v in level0_distribution.items() if k > 0}
    if non_zero_dist:
        labels = [f'{k} classes' for k in sorted(non_zero_dist.keys())]
        sizes = [non_zero_dist[k] for k in sorted(non_zero_dist.keys())]
        colors = plt.cm.Set3(range(len(labels)))
        
        axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 1].set_title('Distribution (Excluding Empty Documents)', fontsize=14)
    
    # Plot 3: Cumulative distribution
    cumulative = []
    total = sum(level0_distribution.values())
    cum_sum = 0
    for k in sorted_keys:
        cum_sum += level0_distribution[k]
        cumulative.append(cum_sum / total * 100)
    
    axes[1, 0].plot(sorted_keys, cumulative, marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Number of Level 0 Classes', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1, 0].set_title('Cumulative Distribution', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(sorted_keys)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1, 0].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    axes[1, 0].legend()
    
    # Plot 4: Histogram (excluding 0)
    non_zero_counts_arr = [c for c in level0_counts if c > 0]
    if non_zero_counts_arr:
        axes[1, 1].hist(non_zero_counts_arr, bins=range(1, max(non_zero_counts_arr)+2), 
                        color='coral', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(non_zero_counts_arr), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(non_zero_counts_arr):.2f}')
        axes[1, 1].axvline(np.median(non_zero_counts_arr), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(non_zero_counts_arr):.1f}')
        axes[1, 1].set_xlabel('Number of Level 0 Classes', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Histogram (Excluding Empty Documents)', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('level0_distribution.png', dpi=150)
    print(f"\n✅ Saved visualization to: level0_distribution.png")
    
    # Sample analysis
    print(f"\n{'='*100}")
    print(f"Sample Documents:")
    print(f"{'='*100}")
    
    # Show samples from each category
    samples_per_category = {}
    for doc_id, classes in core_classes.items():
        if not classes:
            continue
        level0_set = set(class_to_level0.get(int(cid), -1) for cid in classes if class_to_level0.get(int(cid), -1) != -1)
        num_level0 = len(level0_set)
        
        if num_level0 not in samples_per_category:
            samples_per_category[num_level0] = []
        if len(samples_per_category[num_level0]) < 3:
            samples_per_category[num_level0].append((doc_id, classes, level0_set))
    
    for num_level0 in sorted(samples_per_category.keys()):
        print(f"\n--- Documents with {num_level0} Level 0 Classes ---")
        for doc_id, classes, level0_set in samples_per_category[num_level0]:
            print(f"  Doc {doc_id}:")
            print(f"    Core classes: {classes}")
            print(f"    Level 0 classes: {[id2class.get(cid, f'Class {cid}') for cid in level0_set]}")

if __name__ == "__main__":
    analyze_level0_distribution()