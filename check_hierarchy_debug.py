
import sys
import os
sys.path.append('/workspace/20252R0136DATA30400')
from src import utils

def check_hierarchy():
    CLASSES_PATH = "Amazon_products/classes.txt"
    HIERARCHY_PATH = "Amazon_products/class_hierarchy.txt"
    
    id2class, class2id = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, len(id2class))
    
    # Classes to check
    check_ids = [10, 23, 24, 0, 40, 208, 47, 396, 187]
    
    print(f"{'ID':<5} {'Name':<30} {'Parents':<20} {'Children Count'}")
    print("-" * 70)
    
    for cid in check_ids:
        name = id2class.get(cid, "Unknown")
        parents = parents_dict.get(cid, [])
        parent_names = [f"{p}({id2class.get(p, '?')})" for p in parents]
        children = children_dict.get(cid, [])
        
        print(f"{cid:<5} {name:<30} {str(parent_names):<20} {len(children)}")

    # Check if 47 is descendant of 10, 23, 24
    print("\nChecking ancestry for 47 (feminine_care):")
    ancestors = utils.get_ancestors(47, parents_dict)
    print(f"Ancestors of 47: {[f'{a}({id2class[a]})' for a in ancestors]}")

if __name__ == "__main__":
    check_hierarchy()
