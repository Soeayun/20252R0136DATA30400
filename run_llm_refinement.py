"""
Main script to run LLM-based core class refinement
"""

import json
import os
from collections import defaultdict
from src.utils import load_classes, load_hierarchy, load_corpus
from src.llm_refinement import refine_core_classes_with_llm

def main():
    print("=" * 100)
    print("LLM Core Class Refinement - Main Script")
    print("=" * 100)
    
    # API key check is done in llm_refinement.py (with dotenv)
    # Just verify it's loaded
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ ERROR: OPENAI_API_KEY not found!")
        print("Make sure .env file exists with OPENAI_API_KEY=your-key")
        return
    
    # Load data
    print("\n[1] Loading data...")
    id2class, class2id = load_classes("Amazon_products/classes.txt")
    edges = load_hierarchy("Amazon_products/class_hierarchy.txt")
    train_corpus = load_corpus("Amazon_products/train/train_corpus.txt")
    
    print(f"  - Loaded {len(id2class)} classes")
    print(f"  - Loaded {len(edges)} hierarchy edges")
    print(f"  - Loaded {len(train_corpus)} training documents")
    
    # Load existing core classes and ambiguous doc IDs
    print("\n[2] Loading existing core classes and ambiguous doc IDs...")
    with open("checkpoints/core_classes.json", 'r') as f:
        core_classes = json.load(f)
    
    with open("checkpoints/doc_candidates.json", 'r') as f:
        doc_candidates = json.load(f)
    
    # Load pre-filtered ambiguous doc IDs from core_mining
    AMB_IDS_PATH = "checkpoints/ambiguous_doc_ids.json"
    if not os.path.exists(AMB_IDS_PATH):
        print(f"⚠️ ERROR: {AMB_IDS_PATH} not found!")
        print("Please run main.py first to generate core classes and ambiguous doc IDs.")
        return
    
    with open(AMB_IDS_PATH, 'r') as f:
        ambiguous_doc_ids = json.load(f)
    
    print(f"  - Loaded core classes for {len(core_classes)} documents")
    print(f"  - Loaded candidates for {len(doc_candidates)} documents")
    print(f"  - Loaded {len(ambiguous_doc_ids)} ambiguous doc IDs (ratio ≤ 2)")
    
    # Run LLM refinement (no need for Level 0 mapping - direct class selection!)
    print("\n[3] Running LLM refinement...")
    print("  Settings:")
    print(f"    - Max API calls: 1,000")
    print(f"    - Batch size: 10 documents per call")
    print(f"    - Parallel calls: 10 (balanced for rate limits)")
    print(f"    - Target: {len(ambiguous_doc_ids)} ambiguous docs (ratio ≤ 2)")
    print(f"    - Task: Select 0-3 true core classes from up to 10 candidates")
    print(f"    - Expected API calls: {(len(ambiguous_doc_ids) + 9) // 10}")
    print(f"    - Rate limit: 200K TPM (with auto-retry)")
    
    input("\n⏸️  Press Enter to start LLM refinement (this will use API credits)...")
    
    refined_core_classes, llm_decisions = refine_core_classes_with_llm(
        core_classes_dict=core_classes,
        doc_candidates=doc_candidates,
        corpus=train_corpus,
        id2class=id2class,
        ambiguous_doc_ids=ambiguous_doc_ids,
        max_api_calls=1000,
        batch_size=10,
        checkpoint_path="checkpoints/llm_refinement_checkpoint.json"
    )
    
    # Save refined results
    print("\n[6] Saving refined core classes...")
    output_path = "checkpoints/core_classes_llm_refined.json"
    with open(output_path, 'w') as f:
        json.dump(refined_core_classes, f)
    
    print(f"  ✅ Saved to: {output_path}")
    
    # Save LLM decisions for analysis
    decisions_path = "checkpoints/llm_decisions.json"
    with open(decisions_path, 'w') as f:
        json.dump(llm_decisions, f, indent=2)
    
    print(f"  ✅ Saved LLM decisions to: {decisions_path}")
    
    # Analyze basic statistics
    print("\n[7] Basic Statistics:")
    
    # Count class distribution
    class_counts = defaultdict(int)
    total_labels = 0
    empty_docs = 0
    
    for doc_id, classes in refined_core_classes.items():
        num_classes = len(classes) if classes else 0
        class_counts[num_classes] += 1
        total_labels += num_classes
        if num_classes == 0:
            empty_docs += 1
    
    print(f"  Label distribution:")
    for num in sorted(class_counts.keys()):
        count = class_counts[num]
        pct = count / len(refined_core_classes) * 100
        print(f"    {num} labels: {count:>5} docs ({pct:>5.1f}%)")
    
    print(f"\n  Total labels: {total_labels:,}")
    print(f"  Average labels per doc: {total_labels / len(refined_core_classes):.2f}")
    print(f"  Documents with 0 labels: {empty_docs} ({empty_docs/len(refined_core_classes)*100:.1f}%)")
    
    print("\n" + "=" * 100)
    print("✅ LLM Refinement Complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
