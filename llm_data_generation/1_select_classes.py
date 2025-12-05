"""
Step 1: Select classes for synthetic review generation

Analyzes the class hierarchy and selects Level 1 and Level 2 classes
with the appropriate distribution for API budget.

Output: selected_classes.json
"""

import json
import sys
sys.path.append('..')
from src import utils

def load_hierarchy():
    """Load class hierarchy and identify levels"""
    id2class, class2id = utils.load_classes('../Amazon_products/classes.txt')
    parents_dict = {}
    children_dict = {}
    
    with open('../Amazon_products/class_hierarchy.txt', 'r') as f:
        for line in f:
            p, c = map(int, line.strip().split('\t'))
            parents_dict.setdefault(c, []).append(p)
            children_dict.setdefault(p, []).append(c)
    
    return id2class, parents_dict, children_dict

def identify_levels(id2class, parents_dict, children_dict):
    """Identify class levels in hierarchy"""
    # Level 0: Roots (no parents)
    level_0 = [cid for cid in id2class if cid not in parents_dict]
    
    # Level 1: Children of roots
    level_1 = []
    for root in level_0:
        level_1.extend(children_dict.get(root, []))
    level_1 = list(set(level_1))
    
    # Level 2: Children of Level 1
    level_2 = []
    for l1 in level_1:
        level_2.extend(children_dict.get(l1, []))
    level_2 = list(set(level_2))
    
    return level_0, level_1, level_2

def main():
    print("Loading hierarchy...")
    id2class, parents_dict, children_dict = load_hierarchy()
    
    print("Identifying levels...")
    level_0, level_1, level_2 = identify_levels(id2class, parents_dict, children_dict)
    
    print(f"\nHierarchy Statistics:")
    print(f"  Level 0 (Roots): {len(level_0)} classes")
    print(f"  Level 1: {len(level_1)} classes")
    print(f"  Level 2: {len(level_2)} classes")
    
    # Distribution: 0:1:15 ratio for 15,000 reviews
    # TEST: 1024 reviews (~68 API calls, $0.50)
    # PROD: 15000 reviews (~1000 API calls, $3-5)
    total_reviews = 15000  # Minimum: Level1=1개, Level2=2개
    level_1_reviews = int(total_reviews * 1/16)  # 937
    level_2_reviews = int(total_reviews * 15/16)  # 14063
    
    # Reviews per class
    reviews_per_l1 = level_1_reviews // len(level_1) if level_1 else 0
    reviews_per_l2 = level_2_reviews // len(level_2) if level_2 else 0
    
    print(f"\nGeneration Plan:")
    print(f"  Total reviews: {total_reviews}")
    print(f"  Level 1: {level_1_reviews} reviews ({reviews_per_l1} per class)")
    print(f"  Level 2: {level_2_reviews} reviews ({reviews_per_l2} per class)")
    
    # Load keywords
    class2keywords = utils.load_keywords('../Amazon_products/class_related_keywords.txt')
    
    # Build output
    selected_classes = []
    
    # Level 1 classes
    for cid in level_1:
        parent_ids = parents_dict.get(cid, [])
        parent_name = id2class[parent_ids[0]] if parent_ids else "Root"
        
        selected_classes.append({
            "category_id": cid,
            "level": 1,
            "name": id2class[cid],
            "parent_name": parent_name,
            "keywords": class2keywords.get(id2class[cid], [])[:6],
            "num_reviews": reviews_per_l1
        })
    
    # Level 2 classes
    for cid in level_2:
        parent_ids = parents_dict.get(cid, [])
        parent_name = id2class[parent_ids[0]] if parent_ids else "Unknown"
        
        selected_classes.append({
            "category_id": cid,
            "level": 2,
            "name": id2class[cid],
            "parent_name": parent_name,
            "keywords": class2keywords.get(id2class[cid], [])[:6],
            "num_reviews": reviews_per_l2
        })
    
    # Save
    output = {
        "total_classes": len(selected_classes),
        "total_reviews": sum(c['num_reviews'] for c in selected_classes),
        "distribution": {
            "level_1": {"classes": len(level_1), "reviews": level_1_reviews},
            "level_2": {"classes": len(level_2), "reviews": level_2_reviews}
        },
        "classes": selected_classes
    }
    
    with open('selected_classes.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(selected_classes)} classes to selected_classes.json")
    print(f"Estimated API calls: ~{output['total_reviews'] // 15}")

if __name__ == "__main__":
    main()
