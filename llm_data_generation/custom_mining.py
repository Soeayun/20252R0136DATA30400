
import numpy as np
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import utils

def expand_labels_limited(core_classes, parents_dict, children_dict, num_classes):
    """
    Expands labels based on hierarchy with LIMITED scope.
    
    Logic:
    1. Positive: Core Class ONLY.
       - Parents, Grandparents and above are NOT marked as positive (0.0).
    2. Masked: Descendants of Core Class.
       - These are unknown/ignored (0.0 mask).
       
    Returns:
        targets: (num_docs, num_classes) - 1 for Positive, 0 for Negative
        masks: (num_docs, num_classes) - 1 for Valid, 0 for Masked
    """
    print("Expanding Labels (Limited: Core + Immediate Parent)...")
    num_docs = len(core_classes)
    targets = np.zeros((num_docs, num_classes), dtype=np.float32)
    masks = np.ones((num_docs, num_classes), dtype=np.float32)
    
    for i in tqdm(range(num_docs), desc="Expansion"):
        cores = core_classes[i] # List of core class IDs for this doc
        
        # 1. Positives: Core Class ONLY
        positives = set(cores)
        # for c in cores:
        #     # Add immediate parents
        #     parents = parents_dict.get(c, [])
        #     positives.update(parents)
            
        for p in positives:
            targets[i, p] = 1.0
            
        # 2. Masked: Descendants of Core (excluding Core itself)
        descendants = set()
        for c in cores:
            desc = utils.get_descendants(c, children_dict)
            descendants.update(desc)
            
        for d in descendants:
            if d not in positives: 
                masks[i, d] = 0.0
                
    return targets, masks
