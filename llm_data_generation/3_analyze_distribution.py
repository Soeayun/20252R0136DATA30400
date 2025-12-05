"""
Step 3: Analyze the distribution of generated synthetic reviews

Compares actual generated reviews against the intended distribution
from selected_classes.json

Output: Distribution analysis report
"""

import json
from collections import Counter, defaultdict

def analyze_distribution():
    """Analyze and compare review distribution"""
    
    # Load intended distribution
    print("Loading intended distribution...")
    with open('selected_classes.json', 'r') as f:
        intended = json.load(f)
    
    # Load generated reviews
    print("Loading generated reviews...")
    generated_reviews = []
    with open('synthetic_reviews.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    review = json.loads(line)
                    # Handle nested structure
                    if 'product_reviews' in review:
                        generated_reviews.extend(review['product_reviews'])
                    else:
                        generated_reviews.append(review)
                except json.JSONDecodeError:
                    continue
    
    # Count actual reviews per class
    actual_counts = Counter([r['category_id'] for r in generated_reviews if 'category_id' in r])
    
    # Build intended counts dict
    intended_counts = {}
    level_intended = defaultdict(lambda: {'classes': 0, 'reviews': 0})
    
    for cls in intended['classes']:
        cid = cls['category_id']
        level = cls['level']
        num_reviews = cls['num_reviews']
        
        intended_counts[cid] = {
            'name': cls['name'],
            'level': level,
            'intended': num_reviews,
            'actual': actual_counts.get(cid, 0)
        }
        
        level_intended[level]['classes'] += 1
        level_intended[level]['reviews'] += num_reviews
    
    # Calculate actual by level
    level_actual = defaultdict(lambda: {'classes': set(), 'reviews': 0})
    for cid, count in actual_counts.items():
        if cid in intended_counts:
            level = intended_counts[cid]['level']
            level_actual[level]['classes'].add(cid)
            level_actual[level]['reviews'] += count
    
    # Print summary
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal Reviews:")
    print(f"  Intended: {intended['total_reviews']:,}")
    print(f"  Generated: {len(generated_reviews):,}")
    print(f"  Difference: {len(generated_reviews) - intended['total_reviews']:+,}")
    
    print(f"\nBy Level:")
    for level in sorted(level_intended.keys()):
        intended_rev = level_intended[level]['reviews']
        actual_rev = level_actual[level]['reviews']
        intended_cls = level_intended[level]['classes']
        actual_cls = len(level_actual[level]['classes'])
        
        print(f"\n  Level {level}:")
        print(f"    Classes: {actual_cls}/{intended_cls}")
        print(f"    Reviews: {actual_rev:,}/{intended_rev:,} ({actual_rev/intended_rev*100:.1f}%)")
        print(f"    Avg per class: {actual_rev/actual_cls:.1f} (intended: {intended_rev/intended_cls:.1f})")
    
    # Find problematic classes
    print("\n" + "="*70)
    print("PROBLEMATIC CLASSES")
    print("="*70)
    
    missing = []
    under_generated = []
    over_generated = []
    
    for cid, info in intended_counts.items():
        intended_num = info['intended']
        actual_num = info['actual']
        
        if actual_num == 0:
            missing.append((cid, info['name'], intended_num))
        elif actual_num < intended_num * 0.8:  # Less than 80%
            under_generated.append((cid, info['name'], intended_num, actual_num))
        elif actual_num > intended_num * 1.2:  # More than 120%
            over_generated.append((cid, info['name'], intended_num, actual_num))
    
    if missing:
        print(f"\nMissing Classes ({len(missing)}):")
        for cid, name, intended in missing[:10]:
            print(f"  Class {cid} ({name}): 0/{intended} reviews")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
    else:
        print("\n✓ No missing classes")
    
    if under_generated:
        print(f"\nUnder-generated Classes ({len(under_generated)}):")
        for cid, name, intended, actual in under_generated[:10]:
            print(f"  Class {cid} ({name}): {actual}/{intended} ({actual/intended*100:.0f}%)")
        if len(under_generated) > 10:
            print(f"  ... and {len(under_generated)-10} more")
    else:
        print("\n✓ No under-generated classes")
    
    if over_generated:
        print(f"\nOver-generated Classes ({len(over_generated)}):")
        for cid, name, intended, actual in over_generated[:10]:
            print(f"  Class {cid} ({name}): {actual}/{intended} ({actual/intended*100:.0f}%)")
        if len(over_generated) > 10:
            print(f"  ... and {len(over_generated)-10} more")
    else:
        print("\n✓ No over-generated classes")
    
    # Overall quality score
    total_intended = sum(info['intended'] for info in intended_counts.values())
    total_diff = sum(abs(info['actual'] - info['intended']) for info in intended_counts.values())
    quality_score = max(0, 100 - (total_diff / total_intended * 100))
    
    print("\n" + "="*70)
    print(f"OVERALL QUALITY SCORE: {quality_score:.1f}%")
    print("="*70)
    
    if quality_score >= 95:
        print("✓ Excellent distribution!")
    elif quality_score >= 85:
        print("✓ Good distribution")
    elif quality_score >= 70:
        print("⚠ Acceptable distribution (some issues)")
    else:
        print("✗ Poor distribution (significant issues)")
    
    print()

if __name__ == "__main__":
    analyze_distribution()
