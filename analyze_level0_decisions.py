import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def find_level0_ancestor(cid, parents_dict):
    """Find Level 0 ancestor"""
    curr = cid
    while True:
        parents = parents_dict.get(curr, [])
        if not parents:
            return curr
        curr = parents[0]

def analyze_level0_class_counts():
    """Analyze Level 0 CLASS COUNT distribution for decision making"""
    
    # Load data
    print("Loading data...")
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
    
    with open('checkpoints/core_classes.json', 'r') as f:
        core_classes = json.load(f)
    
    # Analyze
    print("Analyzing Level 0 class counts...")
    
    count_ratios = []  # ratio of 1st/2nd count
    first_counts = []
    second_counts = []
    first_percentages = []  # 1st count / total classes
    
    for doc_id, final_classes in tqdm(core_classes.items()):
        if len(final_classes) == 0:
            continue
        
        # Calculate Level 0 class counts
        level0_counts = defaultdict(int)
        
        for c in final_classes:
            lv0 = find_level0_ancestor(int(c), parents_dict)
            level0_counts[lv0] += 1
        
        if len(level0_counts) == 0:
            continue
        
        # Sort by count
        sorted_lv0 = sorted(level0_counts.items(), key=lambda x: x[1], reverse=True)
        total_classes = len(final_classes)
        
        if len(sorted_lv0) >= 1:
            first_counts.append(sorted_lv0[0][1])
            first_percentages.append(sorted_lv0[0][1] / total_classes * 100)
        
        if len(sorted_lv0) >= 2:
            second_counts.append(sorted_lv0[1][1])
            ratio = sorted_lv0[0][1] / sorted_lv0[1][1]
            count_ratios.append(ratio)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Count Ratio Distribution
    axes[0, 0].hist(count_ratios, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='Ratio=2.0')
    axes[0, 0].axvline(3.0, color='orange', linestyle='--', linewidth=2, label='Ratio=3.0')
    axes[0, 0].set_xlabel('Count Ratio (1st / 2nd)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Level 0 Class Count Ratio', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 1st vs 2nd Counts Scatter
    axes[0, 1].scatter(first_counts[:len(second_counts)], second_counts, alpha=0.3, s=10, c='blue')
    axes[0, 1].plot([0, 10], [0, 10], 'r--', linewidth=2, label='Equal')
    axes[0, 1].set_xlabel('1st Place Count', fontsize=12)
    axes[0, 1].set_ylabel('2nd Place Count', fontsize=12)
    axes[0, 1].set_title('1st vs 2nd Count Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 1st Percentage Distribution
    axes[0, 2].hist(first_percentages, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(70, color='red', linestyle='--', linewidth=2, label='70%')
    axes[0, 2].axvline(80, color='orange', linestyle='--', linewidth=2, label='80%')
    axes[0, 2].set_xlabel('1st Place % of Total Classes', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('Dominance: 1st Place Percentage', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Ratio Threshold Impact
    thresholds = np.linspace(1.0, 5.0, 50)
    keep_one_ratios_list = []
    for thresh in thresholds:
        count = sum(1 for ratio in count_ratios if ratio > thresh)
        keep_one_ratios_list.append(count / len(count_ratios) * 100)
    
    axes[1, 0].plot(thresholds, keep_one_ratios_list, linewidth=3, color='purple')
    axes[1, 0].axvline(2.0, color='red', linestyle='--', label='Ratio=2.0')
    axes[1, 0].axvline(3.0, color='orange', linestyle='--', label='Ratio=3.0')
    axes[1, 0].axhline(40, color='green', linestyle='--', alpha=0.5, label='Target 40%')
    axes[1, 0].set_xlabel('Count Ratio Threshold', fontsize=12)
    axes[1, 0].set_ylabel('% Docs → 1 Level 0', fontsize=12)
    axes[1, 0].set_title('Ratio Threshold Impact', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Percentage Threshold Impact
    pct_thresholds = np.linspace(50, 100, 50)
    keep_one_pcts = []
    for thresh in pct_thresholds:
        count = sum(1 for pct in first_percentages if pct > thresh)
        keep_one_pcts.append(count / len(first_percentages) * 100)
    
    axes[1, 1].plot(pct_thresholds, keep_one_pcts, linewidth=3, color='green')
    axes[1, 1].axvline(70, color='red', linestyle='--', label='70%')
    axes[1, 1].axvline(80, color='orange', linestyle='--', label='80%')
    axes[1, 1].axhline(40, color='purple', linestyle='--', alpha=0.5, label='Target 40%')
    axes[1, 1].set_xlabel('1st Place % Threshold', fontsize=12)
    axes[1, 1].set_ylabel('% Docs → 1 Level 0', fontsize=12)
    axes[1, 1].set_title('Percentage Threshold Impact', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Rule Comparison
    rules_data = []
    rule_names = []
    
    count1 = sum(1 for ratio in count_ratios if ratio > 2.0)
    rules_data.append(count1 / len(count_ratios) * 100)
    rule_names.append('Ratio>2.0')
    
    count2 = sum(1 for ratio in count_ratios if ratio > 3.0)
    rules_data.append(count2 / len(count_ratios) * 100)
    rule_names.append('Ratio>3.0')
    
    count3 = sum(1 for pct in first_percentages if pct > 70)
    rules_data.append(count3 / len(first_percentages) * 100)
    rule_names.append('1st>70%')
    
    count4 = sum(1 for pct in first_percentages if pct > 80)
    rules_data.append(count4 / len(first_percentages) * 100)
    rule_names.append('1st>80%')
    
    count5 = sum(1 for i in range(len(count_ratios))
                 if count_ratios[i] > 2.0 or first_percentages[i] > 70)
    rules_data.append(count5 / len(count_ratios) * 100)
    rule_names.append('Combined')
    
    axes[1, 2].bar(rule_names, rules_data, 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'yellow', 'pink'], 
                   edgecolor='black', linewidth=2)
    axes[1, 2].set_ylabel('% Docs → 1 Level 0', fontsize=12)
    axes[1, 2].set_title('Rule Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(rules_data):
        axes[1, 2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('level0_decision_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Analysis saved to: level0_decision_analysis.png")
    
    # Print Statistics
    print("\n" + "="*80)
    print("LEVEL 0 CLASS COUNT ANALYSIS")
    print("="*80)
    print(f"Total documents with 2+ Level 0: {len(count_ratios)}")
    
    print(f"\nCount Ratio Statistics (1st / 2nd):")
    print(f"  Mean: {np.mean(count_ratios):.2f}")
    print(f"  Median: {np.median(count_ratios):.2f}")
    print(f"  Std: {np.std(count_ratios):.2f}")
    print(f"  Min: {np.min(count_ratios):.2f}")
    print(f"  Max: {np.max(count_ratios):.2f}")
    
    print(f"\n1st Place Count:")
    print(f"  Mean: {np.mean(first_counts):.1f}")
    print(f"  Median: {np.median(first_counts):.1f}")
    
    print(f"\n2nd Place Count:")
    print(f"  Mean: {np.mean(second_counts):.1f}")
    print(f"  Median: {np.median(second_counts):.1f}")
    
    print(f"\n1st Place Percentage:")
    print(f"  Mean: {np.mean(first_percentages):.1f}%")
    print(f"  Median: {np.median(first_percentages):.1f}%")
    
    print(f"\n" + "="*80)
    print("DECISION RULES IMPACT")
    print("="*80)
    print(f"Rule 1 (Ratio > 2.0):        {rules_data[0]:.1f}% → 1 Level 0")
    print(f"Rule 2 (Ratio > 3.0):        {rules_data[1]:.1f}% → 1 Level 0")
    print(f"Rule 3 (1st > 70%):          {rules_data[2]:.1f}% → 1 Level 0")
    print(f"Rule 4 (1st > 80%):          {rules_data[3]:.1f}% → 1 Level 0")
    print(f"Rule 5 (Combined OR):        {rules_data[4]:.1f}% → 1 Level 0")
    print("="*80)

if __name__ == "__main__":
    analyze_level0_class_counts()
