"""
LLM-based Core Class Selection from Candidates

Goal: Select 0-3 true core classes from up to 10 candidates
Uses async parallel API calls for maximum speed.
"""

import json
import os
from tqdm import tqdm
import openai
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError(
        "❌ OPENAI_API_KEY not found!\n"
        "Please create a .env file with:\n"
        "OPENAI_API_KEY=your-key-here\n"
        "Or set it as environment variable: export OPENAI_API_KEY='your-key'"
    )

API_MODEL = "gpt-4o-mini"
MAX_API_CALLS = 1000
BATCH_SIZE = 25
MAX_PARALLEL_CALLS = 10  # 10 parallel calls to balance speed and rate limits


def create_llm_prompt(doc_texts: List[str], 
                     candidates_list: List[List[tuple]],
                     ) -> str:
    """
    Create prompt for LLM to select true core classes from candidates
    
    Returns JSON format request
    """
    
    prompt = """You are an expert in product categorization. For each product review below, select the TRUE core classes that the review is genuinely about.

**STRICT REQUIREMENTS:**
1. Select MAXIMUM 3 classes per review (can be 0, 1, 2, or 3)
2. NEVER exceed 3 classes - this is a hard constraint
3. Be conservative: only select classes you're highly confident about
4. Consider hierarchy: if selecting "baby formula", also include "baby food" and "baby products"
5. If uncertain about all candidates, select 0 classes

**Instructions:**
- Read each review carefully
- From the candidate classes provided, identify the most relevant ones
- Select 0-3 classes (0 if none are truly relevant, up to 3 maximum)
- Quality over quantity: better to select fewer accurate classes than many uncertain ones

Respond in JSON format with EXACTLY this structure:
{
    "selections": [
        {"doc_id": <id>, "selected_class_ids": [id1, id2, id3], "reasoning": "brief explanation"},
        ...
    ]
}

**CRITICAL: Each "selected_class_ids" array MUST contain 0 to 3 integers only. No more than 3!**

Reviews and Candidates:
"""
    
    for i, (doc_text, candidates) in enumerate(zip(doc_texts, candidates_list)):
        # Truncate review
        truncated_text = doc_text[:800] + ("..." if len(doc_text) > 800 else "")
        
        prompt += f"""
---
Document {i+1}:
Review: {truncated_text}

Candidate Classes (select 0-3 most relevant, MAXIMUM 3):
"""
        for cid, cname, score in candidates:
            prompt += f"  - ID {cid}: {cname} (score: {score:.3f})\n"
    
    return prompt


def call_llm_batch_sync(doc_batch: List[tuple]) -> Dict:
    """
    Synchronous LLM API call for a batch of documents
    
    Args:
        doc_batch: List of (doc_id, doc_text, [(class_id, class_name, score), ...])
    
    Returns:
        {doc_id: [selected_class_ids]}
    """
    doc_ids = [item[0] for item in doc_batch]
    doc_texts = [item[1] for item in doc_batch]
    candidates_list = [item[2] for item in doc_batch]
    
    prompt = create_llm_prompt(doc_texts, candidates_list)
    
    try:
        # Use new OpenAI client (1.0+)
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=[
                {"role": "system", "content": "You are a product categorization expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Map results back to doc_ids
        selections = {}
        for i, selection in enumerate(result.get("selections", [])):
            if i < len(doc_ids):
                selected_ids = selection.get("selected_class_ids", [])
                # Ensure 0-3 classes
                if len(selected_ids) > 3:
                    selected_ids = selected_ids[:3]
                
                selections[doc_ids[i]] = selected_ids
        
        return selections
    
    except Exception as e:
        print(f"\n❌ LLM API Error: {e}")
        print("Stopping execution due to API failure.")
        raise SystemExit(1)


async def call_llm_batch_async(doc_batch: List[tuple], executor: ThreadPoolExecutor) -> Dict:
    """
    Async wrapper for parallel LLM API calls
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, call_llm_batch_sync, doc_batch)


def refine_core_classes_with_llm(
    core_classes_dict: Dict,
    doc_candidates: Dict,
    corpus: Dict,
    id2class: Dict,
    ambiguous_doc_ids: List[int],
    max_api_calls: int = MAX_API_CALLS,
    batch_size: int = BATCH_SIZE,
    max_parallel: int = MAX_PARALLEL_CALLS,
    checkpoint_path: str = "checkpoints/llm_refinement_checkpoint.json"
):
    """
    Main function: Use LLM to select true core classes from candidates (PARALLEL)
    
    Args:
        ambiguous_doc_ids: Docs needing LLM judgment (ratio <= 2 cases)
        max_parallel: Maximum number of parallel API calls
    """
    
    print("=" * 100)
    print("LLM-based Core Class Selection (Parallel Processing)")
    print("=" * 100)
    
    # Load checkpoint
    processed_docs = set()
    llm_selections = {}
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            processed_docs = set(checkpoint.get('processed_docs', []))
            llm_selections = checkpoint.get('selections', {})
        print(f"Resumed: {len(processed_docs)} docs already processed")
    
    # Prepare ambiguous docs data
    print(f"\n[Step 1] Received {len(ambiguous_doc_ids)} ambiguous documents")
    
    ambiguous_docs_data = []
    for doc_id in ambiguous_doc_ids:
        if doc_id in processed_docs:
            continue
        
        # Get candidates for this doc
        core_classes = core_classes_dict.get(str(doc_id), core_classes_dict.get(doc_id, []))
        if not core_classes:
            continue
        
        # Build candidate list with scores
        candidates = []
        for cid in core_classes:
            cname = id2class.get(cid, f"Class_{cid}").replace('_', ' ')
            score = doc_candidates.get(str(doc_id), {}).get(str(cid), 0.0)
            candidates.append((cid, cname, score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        doc_text = corpus.get(doc_id, "")
        ambiguous_docs_data.append((doc_id, doc_text, candidates))
    
    print(f"Remaining to process: {len(ambiguous_docs_data)}")
    
    if len(ambiguous_docs_data) == 0:
        print("No documents to process!")
        return core_classes_dict, llm_selections
    
    # Process with LLM (PARALLEL)
    print(f"\n[Step 2] Processing with LLM (Budget: {max_api_calls} calls, Batch: {batch_size}, Parallel: {max_parallel})...")
    
    # Split into batches
    batches = []
    for i in range(0, len(ambiguous_docs_data), batch_size):
        batch = ambiguous_docs_data[i:i+batch_size]
        batches.append(batch)
    
    # Limit by max_api_calls
    batches = batches[:max_api_calls]
    
    print(f"Total batches to process: {len(batches)}")
    
    # Process batches in parallel
    start_time = time.time()
    
    async def process_all_batches():
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
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
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    for result in all_results:
        llm_selections.update({str(k): v for k, v in result.items()})
    
    # Update processed docs
    for batch in batches:
        processed_docs.update([item[0] for item in batch])
    
    # Save final checkpoint
    checkpoint = {
        'processed_docs': list(processed_docs),
        'selections': llm_selections
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)
    
    print(f"\n⏱️  Processing time: {elapsed:.1f}s ({len(batches)/elapsed:.1f} batches/sec)")
    
    # Apply LLM selections
    print(f"\n[Step 3] Applying LLM selections...")
    
    refined_core_classes = {}
    for doc_id, core_classes in core_classes_dict.items():
        doc_id_str = str(doc_id)
        
        if doc_id_str in llm_selections:
            # Use LLM selection
            refined_core_classes[doc_id] = llm_selections[doc_id_str]
        else:
            # Keep original (not ambiguous or not processed)
            refined_core_classes[doc_id] = core_classes
    
    print(f"\n✅ Refinement Complete!")
    print(f"   API calls made: {len(batches)}")
    print(f"   Documents refined: {len(llm_selections)}")
    print(f"   Average time per call: {elapsed/len(batches):.2f}s")
    
    return refined_core_classes, llm_selections
