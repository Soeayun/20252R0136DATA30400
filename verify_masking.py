
import numpy as np
import sys
import os

# Mock utils and core_mining functions to avoid importing the whole project if possible, 
# but better to import the actual module to test the real code.
sys.path.append('/workspace/20252R0136DATA30400')
from src import core_mining

def test_masking():
    # Define a simple hierarchy: 0 -> 1 (0 is parent of 1)
    # Total classes: 3 (0, 1, 2) where 2 is unrelated.
    
    num_classes = 3
    parents_dict = {
        0: [],
        1: [0],
        2: []
    }
    children_dict = {
        0: [1],
        1: [],
        2: []
    }
    
    # Case: Document labeled as class 0 (Parent)
    # Expected: 
    # - Target 0: 1.0 (Positive)
    # - Target 1: 0.0 (Not positive)
    # - Mask 0: 1.0 (Valid)
    # - Mask 1: 0.0 (MASKED - because it's a descendant of a positive class 0)
    # - Mask 2: 1.0 (Valid - negative)
    
    core_classes = [[0]]
    
    print("Running expand_labels...")
    targets, masks = core_mining.expand_labels(core_classes, parents_dict, children_dict, num_classes)
    
    print(f"Targets: {targets[0]}")
    print(f"Masks:   {masks[0]}")
    
    # Verification
    assert targets[0, 0] == 1.0, "Class 0 should be positive"
    assert targets[0, 1] == 0.0, "Class 1 should not be positive (it's unknown)"
    assert masks[0, 0] == 1.0, "Class 0 should be unmasked"
    assert masks[0, 1] == 0.0, "Class 1 should be MASKED (0.0)"
    assert masks[0, 2] == 1.0, "Class 2 should be unmasked (negative)"
    
    print("\nSUCCESS: Masking logic is working correctly!")

if __name__ == "__main__":
    test_masking()
