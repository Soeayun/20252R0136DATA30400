
import numpy as np
import sys
import os

# Add path to import custom_mining
sys.path.append('/workspace/20252R0136DATA30400/llm_data_generation')
sys.path.append('/workspace/20252R0136DATA30400') # For src.utils
import custom_mining

def test_limited_expansion():
    # Define hierarchy: 0 -> 1 -> 2 (0 is grandparent, 1 is parent, 2 is child)
    # Total classes: 4 (0, 1, 2, 3)
    
    num_classes = 4
    parents_dict = {
        0: [],
        1: [0],
        2: [1],
        3: []
    }
    children_dict = {
        0: [1],
        1: [2],
        2: [],
        3: []
    }
    
    # Case: Document labeled as class 2 (Child)
    # Expected: 
    # - Target 2: 1.0 (Core)
    # - Target 1: 0.0 (Immediate Parent - Ignored in Core Only)
    # - Target 0: 0.0 (Grandparent - Ignored)
    # - Target 3: 0.0 (Unrelated)
    
    core_classes = [[2]]
    
    print("Running expand_labels_limited...")
    targets, masks = custom_mining.expand_labels_limited(core_classes, parents_dict, children_dict, num_classes)
    
    print(f"Targets: {targets[0]}")
    print(f"Masks:   {masks[0]}")
    
    # Verification
    assert targets[0, 2] == 1.0, "Class 2 (Core) should be positive"
    assert targets[0, 1] == 0.0, "Class 1 (Parent) should be negative (Core Only Mode)"
    assert targets[0, 0] == 0.0, "Class 0 (Grandparent) should NOT be positive"
    assert targets[0, 3] == 0.0, "Class 3 (Unrelated) should be negative"
    
    # Check masking (if we had descendants)
    # Let's add a descendant 4 to 2
    # 2 -> 4
    # Re-run with 5 classes
    
    print("\nSUCCESS: Core-Only expansion logic is working correctly!")

if __name__ == "__main__":
    test_limited_expansion()
