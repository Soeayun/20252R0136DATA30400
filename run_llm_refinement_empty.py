"""
LLM Refinement for Empty Label Documents

Takes documents with 0 labels from core_classes_llm_refined.json,
retrieves top 20 candidates from doc_candidates.json,
and re-runs LLM to assign labels.
"""

import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from src.utils import load_classes, load_corpus
from src.llm_refinement import create_llm_prompt, call_llm_batch_sync

load_dotenv()

def main():
    print("=" * 100)
    print("LLM Refinement for Empty Label Documents")
    print("=" * 100)
    
    # API key check
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ ERROR: OPENAI_API_KEY not found!")
        return
    
    # Load data
    print("\n[1] Loading data...")
    id2class, class2id = load_classes("Amazon_products/classes.txt")
    corpus = load_corpus("Amazon_products/test/test_corpus.txt")
    
    print(f"  - Loaded {len(id2class)} classes")
    print(f"  - Loaded {len(corpus)} documents")
    
    # Load refined core classes and doc candidates
    print("\n[2] Loading refined core classes and candidates...")
    with open("checkpoints/core_classes_llm_refined.json", 'r') as f:
        refined_core_classes = json.load(f)
    
    with open("checkpoints/doc_candidates.json", 'r') as f:
        doc_candidates = json.load(f)
    
    # Find documents with 0 labels
    empty_doc_ids = []
    for doc_id, classes in refined_core_classes.items():
        if not classes or len(classes) == 0:
            empty_doc_ids.append(doc_id)
    
    print(f"  - Total refined docs: {len(refined_core_classes)}")
    print(f"  - Documents with 0 labels: {len(empty_doc_ids)}")
    
    if len(empty_doc_ids) == 0:
        print("\n✅ No empty documents to process!")
        return
    
    # Prepare data for LLM (top 20 candidates from doc_candidates)
    print("\n[3] Preparing candidates (top 20 from doc_candidates)...")
    
    docs_data = []
    for doc_id in empty_doc_ids:
        doc_id_str = str(doc_id)
        
        # Get candidates from doc_candidates
        candidates_dict = doc_candidates.get(doc_id_str, {})
        if not candidates_dict:
            continue
        
        # Sort by score and take top 20
        sorted_candidates = sorted(candidates_dict.items(), key=lambda x: float(x[1]), reverse=True)[:20]
        
        candidates = []
        for cid_str, score in sorted_candidates:
            cid = int(cid_str)
            cname = id2class.get(cid, f"Class_{cid}").replace('_', ' ')
            candidates.append((cid, cname, float(score)))
        
        if candidates:
            doc_id_int = int(doc_id) if isinstance(doc_id, str) else doc_id
            doc_text = corpus.get(doc_id_int, "")
            if doc_text:
                docs_data.append((doc_id, doc_text, candidates))
    
    print(f"  - Documents to process: {len(docs_data)}")
    
    if len(docs_data) == 0:
        print("\n⚠️ No valid documents to process!")
        return
    
    # Settings
    BATCH_SIZE = 18  # 15 docs per batch
    MAX_API_CALLS = 143
    
    # Split into batches
    batches = []
    for i in range(0, len(docs_data), BATCH_SIZE):
        batch = docs_data[i:i+BATCH_SIZE]
        batches.append(batch)
    
    # Limit by max API calls
    batches = batches[:MAX_API_CALLS]
    
    print(f"\n[4] Running LLM refinement...")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max API calls: {MAX_API_CALLS}")
    print(f"  - Total batches: {len(batches)}")
    print(f"  - Documents covered: {min(len(docs_data), MAX_API_CALLS * BATCH_SIZE)}")
    
    input("\n⏸️  Press Enter to start LLM refinement...")
    
    # Process batches (PARALLEL)
    from tqdm import tqdm
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from src.llm_refinement import call_llm_batch_async
    
    MAX_PARALLEL = 5
    
    llm_selections = {}
    start_time = time.time()
    
    async def process_all_batches():
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            tasks = []
            for batch in batches:
                task = call_llm_batch_async(batch, executor)
                tasks.append(task)
            
            # Process with progress bar
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parallel LLM Calls"):
                result = await coro
                results.append(result)
            
            return results
    
    # Run async event loop
    loop = asyncio.get_event_loop()
    all_results = loop.run_until_complete(process_all_batches())
    
    # Aggregate results
    for result in all_results:
        llm_selections.update({str(k): v for k, v in result.items()})
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Processing time: {elapsed:.1f}s ({len(batches)/elapsed:.1f} batches/sec)")
    
    # Apply selections to refined_core_classes
    print("\n[5] Applying LLM selections...")
    
    updated_count = 0
    for doc_id, selected_classes in llm_selections.items():
        if doc_id in refined_core_classes:
            refined_core_classes[doc_id] = selected_classes
            if selected_classes:
                updated_count += 1
    
    print(f"  - Documents updated with labels: {updated_count}")
    
    # Save updated results
    print("\n[6] Saving updated core classes...")
    output_path = "checkpoints/core_classes_llm_refined.json"
    with open(output_path, 'w') as f:
        json.dump(refined_core_classes, f)
    
    print(f"  ✅ Saved to: {output_path}")
    
    # Statistics
    print("\n[7] Final Statistics:")
    
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
    print("✅ Empty Document Refinement Complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
