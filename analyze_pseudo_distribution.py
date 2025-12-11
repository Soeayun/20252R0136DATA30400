"""
Analyze pseudo-label distribution from test corpus iterations
"""

import json
import numpy as np
from collections import Counter

def analyze_pseudo_labels(json_path):
    """
    Analyze pseudo-label confidence distribution
    
    Args:
        json_path: Path to pseudo_labels JSON file
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {json_path}")
    print(f"{'='*80}")
    
    # Load data
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # Statistics
    total_docs = len(predictions)
    accepted_docs = sum(1 for pred in predictions.values() if pred['accepted'])
    rejected_docs = total_docs - accepted_docs
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total documents: {total_docs:,}")
    print(f"   Accepted: {accepted_docs:,} ({accepted_docs/total_docs*100:.1f}%)")
    print(f"   Rejected: {rejected_docs:,} ({rejected_docs/total_docs*100:.1f}%)")
    
    # Analyze accepted documents
    if accepted_docs > 0:
        accepted_preds = [pred for pred in predictions.values() if pred['accepted']]
        
        # Confidence distribution
        all_confidences = []
        avg_confidences = []
        num_classes_dist = []
        
        for pred in accepted_preds:
            confidences = pred['confidences']
            all_confidences.extend(confidences)
            avg_confidences.append(np.mean(confidences))
            num_classes_dist.append(pred['num_classes'])
        
        print(f"\n📈 Confidence Distribution (Accepted):")
        print(f"   All predictions:")
        print(f"      Mean: {np.mean(all_confidences):.4f}")
        print(f"      Median: {np.median(all_confidences):.4f}")
        print(f"      Std: {np.std(all_confidences):.4f}")
        print(f"      Min: {np.min(all_confidences):.4f}")
        print(f"      Max: {np.max(all_confidences):.4f}")
        
        print(f"\n   Average per document:")
        print(f"      Mean: {np.mean(avg_confidences):.4f}")
        print(f"      Median: {np.median(avg_confidences):.4f}")
        print(f"      Std: {np.std(avg_confidences):.4f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n   Percentiles:")
        for p in percentiles:
            val = np.percentile(avg_confidences, p)
            print(f"      {p}th: {val:.4f}")
        
        # Confidence ranges
        print(f"\n   Distribution by range:")
        ranges = [
            (0.0, 0.8, "< 0.8 (Low)"),
            (0.8, 0.85, "0.8-0.85"),
            (0.85, 0.90, "0.85-0.90"),
            (0.90, 0.95, "0.90-0.95"),
            (0.95, 0.99, "0.95-0.99"),
            (0.99, 1.0, "≥ 0.99 (Very High)")
        ]
        
        for low, high, label in ranges:
            count = sum(1 for c in avg_confidences if low <= c < high or (high == 1.0 and c == 1.0))
            pct = count / len(avg_confidences) * 100 if avg_confidences else 0
            print(f"      {label}: {count:4d} ({pct:5.1f}%)")
        
        # Number of classes distribution
        print(f"\n🏷️  Number of Classes Distribution:")
        class_counter = Counter(num_classes_dist)
        for num_classes in sorted(class_counter.keys()):
            count = class_counter[num_classes]
            pct = count / len(num_classes_dist) * 100
            print(f"      {num_classes} classes: {count:4d} ({pct:5.1f}%)")
        
        print(f"      Mean: {np.mean(num_classes_dist):.2f}")
        print(f"      Median: {np.median(num_classes_dist):.1f}")
        
        # Limited to top-3 analysis
        limited_count = sum(1 for pred in accepted_preds if pred.get('limited_to_top3', False))
        if limited_count > 0:
            print(f"\n⚠️  Limited to top-3: {limited_count:,} documents ({limited_count/accepted_docs*100:.1f}%)")
    
    else:
        print("\n⚠️  No documents were accepted!")
    
    return {
        'total': total_docs,
        'accepted': accepted_docs,
        'rejected': rejected_docs,
        'avg_confidences': avg_confidences if accepted_docs > 0 else []
    }


def compare_iterations(json_paths):
    """
    Compare multiple iterations
    
    Args:
        json_paths: List of paths to pseudo_labels JSON files
    """
    print(f"\n{'='*80}")
    print(f"Comparing Multiple Iterations")
    print(f"{'='*80}\n")
    
    results = []
    for path in json_paths:
        try:
            result = analyze_pseudo_labels(path)
            results.append((path, result))
        except FileNotFoundError:
            print(f"\n⚠️  File not found: {path}")
    
    # Summary table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"Summary Comparison")
        print(f"{'='*80}\n")
        print(f"{'Iteration':<20} {'Accepted':>10} {'Acceptance Rate':>15} {'Mean Conf':>12}")
        print(f"{'-'*60}")
        
        for path, result in results:
            iter_name = path.split('/')[-1].replace('pseudo_labels_', '').replace('.json', '')
            rate = result['accepted'] / result['total'] * 100 if result['total'] > 0 else 0
            mean_conf = np.mean(result['avg_confidences']) if result['avg_confidences'] else 0
            print(f"{iter_name:<20} {result['accepted']:>10,} {rate:>14.1f}% {mean_conf:>12.4f}")


if __name__ == "__main__":
    import sys
    import os
    
    # Default: analyze test iterations
    if len(sys.argv) > 1:
        # Specific file provided
        json_path = sys.argv[1]
        analyze_pseudo_labels(json_path)
    else:
        # Analyze all test iterations
        checkpoint_dir = "checkpoints"
        test_files = []
        
        for i in range(1, 10):
            path = os.path.join(checkpoint_dir, f"pseudo_labels_test_iter{i}.json")
            if os.path.exists(path):
                test_files.append(path)
        
        if test_files:
            compare_iterations(test_files)
        else:
            print("No pseudo_labels_test_iter*.json files found in checkpoints/")
            print("\nUsage: python analyze_pseudo_distribution.py [json_file]")
