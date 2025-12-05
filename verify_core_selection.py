import unittest
import numpy as np
import sys
import os
from collections import defaultdict

# Mocking necessary structures
class MockUtils:
    @staticmethod
    def get_siblings(c, parents_dict, children_dict):
        # Simplified sibling logic for testing
        parents = parents_dict.get(c, [])
        siblings = set()
        for p in parents:
            siblings.update(children_dict.get(p, []))
        if c in siblings:
            siblings.remove(c)
        return list(siblings)

    @staticmethod
    def get_ancestors(c, parents_dict):
        ancestors = set()
        curr = c
        while True:
            parents = parents_dict.get(curr, [])
            if not parents:
                break
            curr = parents[0]
            ancestors.add(curr)
        return ancestors

# We need to import the function to test. 
# Since we can't easily import just the function if it depends on other things, 
# we will copy the logic to be tested or mock the module.
# Ideally, we import the module. Let's try to import src.core_mining.
# We need to mock src.utils inside core_mining if possible, or just use the real one if it's pure logic.
# src.utils seems to be pure logic for hierarchy.

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import core_mining

class TestCoreSelection(unittest.TestCase):
    def setUp(self):
        # Define a simple hierarchy
        # Level 0: 100, 200
        # Level 1: 110 (child of 100), 210 (child of 200)
        # Level 2: 111 (child of 110), 211 (child of 210)
        self.parents_dict = {
            100: [], 200: [],
            110: [100], 210: [200],
            111: [110], 211: [210]
        }
        self.children_dict = {
            100: [110], 200: [210],
            110: [111], 210: [211],
            111: [], 211: []
        }
        
    def test_relative_drop_and_max_k(self):
        # Mock doc_candidates: {doc_id: {class_id: score}}
        # Scenario: 15 candidates.
        # Scores drop sharply after index 2 (0.9, 0.85, 0.4, ...)
        
        # We need to mock the internal logic of identify_confident_core_classes
        # But that function does a lot. 
        # Let's create a wrapper or just test the logic if we extract it.
        # Since I am modifying the function in place, I will test the modified function.
        
        # Construct input
        doc_candidates = {
            0: {
                111: 0.9, # L0=100
                110: 0.85, # L0=100
                211: 0.4, # L0=200 (Drop > 0.1 from previous?) No, 0.85 - 0.4 = 0.45 > 0.1. STOP here?
                # Wait, the relative drop is between sorted candidates.
                # Sorted: 0.9, 0.85, 0.4.
                # 0.9 - 0.85 = 0.05 (Keep)
                # 0.85 - 0.4 = 0.45 (Drop! Stop)
                # So should only keep 111, 110.
            }
        }
        
        # Run function
        # Note: The function also calculates confidence scores internally.
        # Conf = Score - Max(Parent, Sibling)
        # 111: Parent 110 (0.85). Conf = 0.9 - 0.85 = 0.05
        # 110: Parent 100 (0.0). Conf = 0.85 - 0 = 0.85
        # 211: Parent 210 (0.0). Conf = 0.4 - 0 = 0.4
        
        # Sorted Confs:
        # 110: 0.85
        # 211: 0.40
        # 111: 0.05
        
        # Relative Drop Check:
        # 1. 110 (0.85). Next is 211 (0.40). Diff = 0.45 > 0.1.
        # STOP.
        # Selected: [110]
        
        # Level 0 Selection:
        # 110 -> L0=100. Score = 0.85.
        # Result: {0: [110]}
        
        result = core_mining.identify_confident_core_classes(
            doc_candidates, self.parents_dict, self.children_dict
        )
        
        self.assertEqual(result[0], [110])

    def test_level0_ratio_filtering(self):
        # Scenario: Two strong topics.
        # 110 (L0=100): Conf 0.9
        # 210 (L0=200): Conf 0.8
        # 110 - 210 = 0.1 (Not > 0.1, so keep both)
        
        doc_candidates = {
            0: {
                110: 0.9,
                210: 0.8
            }
        }
        # Confs: 110=0.9, 210=0.8
        
        # Level 0 Scores:
        # L0=100: 0.9
        # L0=200: 0.8
        # Ratio: 0.8 / 0.9 = 0.88 >= 0.6. Keep both.
        
        result = core_mining.identify_confident_core_classes(
            doc_candidates, self.parents_dict, self.children_dict
        )
        self.assertCountEqual(result[0], [110, 210])
        
    def test_level0_ratio_filtering_drop(self):
        # Scenario: One strong, one weak.
        # 110 (L0=100): Conf 0.9
        # 210 (L0=200): Conf 0.4
        # Diff 0.5 > 0.1. STOP?
        # Wait, if Relative Drop kicks in, 210 is removed before Level 0 check.
        # We need a case where Relative Drop passes but Level 0 fails?
        # Relative Drop is local (i vs i+1).
        # 110 (0.9), 111 (0.85), 210 (0.4)
        # 0.9 - 0.85 = 0.05 (Keep)
        # 0.85 - 0.4 = 0.45 (Stop)
        # So 210 is lost anyway.
        
        # What if scores are close but L0 sums are different?
        # 110 (0.9), 111 (0.85) -> L0=100 Sum = 1.75
        # 210 (0.8) -> L0=200 Sum = 0.8
        # Sorted: 110(0.9), 111(0.85), 210(0.8)
        # Diffs: 0.05, 0.05. All kept.
        
        # L0 Scores:
        # 100: 1.75
        # 200: 0.8
        # Ratio: 0.8 / 1.75 = 0.45 < 0.6.
        # L0=200 should be dropped.
        
        doc_candidates = {
            0: {
                110: 0.9,
                111: 0.85, # Parent 110 is in cand? 
                # If 110 is in cand, 111 Conf = 0.85 - 0.9 = -0.05.
                # So 111 will be very low rank.
                # Let's make them siblings or unrelated.
                # 110 (L0=100), 120 (L0=100)
                # 210 (L0=200)
            }
        }
        # Update hierarchy for this test to avoid sibling penalty
        # 100 -> 101 -> 110
        # 100 -> 102 -> 120
        # 200 -> 210
        self.parents_dict[101] = [100]
        self.parents_dict[102] = [100]
        self.parents_dict[110] = [101] # Was [100]
        self.parents_dict[120] = [102] # Was [100]
        
        self.children_dict[100] = [101, 102] # Was [110, 120]
        self.children_dict[101] = [110]
        self.children_dict[102] = [120]
        
        doc_candidates = {
            0: {
                110: 0.9,
                120: 0.85,
                210: 0.8
            }
        }
        # Confs: 
        # 110: Parent 101 (not in cand). Sibling (none). Conf = 0.9
        # 120: Parent 102 (not in cand). Sibling (none). Conf = 0.85
        # 210: Parent 200 (not in cand). Sibling (none). Conf = 0.8
        
        # Sorted: 110(0.9), 120(0.85), 210(0.8)
        # Diffs: 0.05, 0.05. All kept by Relative Drop.
        
        # L0 Scores:
        # 100: 0.9 + 0.85 = 1.75
        # 200: 0.8
        # Ratio: 0.8 / 1.75 = 0.457 < 0.6. Drop 200.
        
        result = core_mining.identify_confident_core_classes(
            doc_candidates, self.parents_dict, self.children_dict
        )
        self.assertCountEqual(result[0], [110, 120])

if __name__ == '__main__':
    unittest.main()
