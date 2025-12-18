"""
LLM-based Pseudo Labeling with Hierarchy Paths

Goal: Select exactly ONE most specific category (with full hierarchy path) or NONE
Uses async parallel API calls for maximum speed.
Automatically expands selected class to include all ancestor classes.
"""

import json
import os
from tqdm import tqdm
import openai
from typing import Dict, List, Tuple, Optional
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
MAX_PARALLEL_CALLS = 5  # 10 parallel calls to balance speed and rate limits


# ============ Hierarchy Path Utilities ============

def build_parent_mapping(edges: List[Tuple[int, int]]) -> Dict[int, int]:
    """
    Build child -> parent mapping from edges
    edges: List of (parent_id, child_id)
    """
    child_to_parent = {}
    for parent, child in edges:
        child_to_parent[child] = parent
    return child_to_parent


def get_ancestors(class_id: int, child_to_parent: Dict[int, int]) -> List[int]:
    """
    Get all ancestor class IDs (including self), from leaf to root
    Returns: [class_id, parent_id, grandparent_id, ...]
    """
    ancestors = [class_id]
    current = class_id
    while current in child_to_parent:
        parent = child_to_parent[current]
        ancestors.append(parent)
        current = parent
    return ancestors


def build_hierarchy_path(class_id: int, child_to_parent: Dict[int, int], id2class: Dict) -> str:
    """
    Build full hierarchy path string for a class
    Example: "grocery_gourmet_food > meat_poultry > jerky"
    """
    ancestors = get_ancestors(class_id, child_to_parent)
    # Reverse to get root -> leaf order
    ancestors = ancestors[::-1]
    
    path_names = []
    for cid in ancestors:
        name = id2class.get(cid, f"Class_{cid}").replace('_', ' ')
        path_names.append(name)
    
    return " > ".join(path_names)


def get_hierarchy_level(class_id: int, child_to_parent: Dict[int, int]) -> int:
    """
    Get the hierarchy level of a class (0 = root, 1 = child of root, etc.)
    """
    level = 0
    current = class_id
    while current in child_to_parent:
        level += 1
        current = child_to_parent[current]
    return level


# ============ Global hierarchy data (set by main function) ============
_child_to_parent: Dict[int, int] = {}
_id2class: Dict[int, str] = {}


def set_hierarchy_data(edges: List[Tuple[int, int]], id2class: Dict[int, str]):
    """Set global hierarchy data for use in prompts"""
    global _child_to_parent, _id2class
    _child_to_parent = build_parent_mapping(edges)
    _id2class = id2class


def create_llm_prompt(doc_texts: List[str], 
                     candidates_list: List[List[tuple]],
                     ) -> str:
    """
    Create prompt for LLM to select exactly ONE category path or NONE
    Shows full hierarchy paths for each candidate
    """
    
    prompt = """You are an expert in product categorization. For each product review below, identify the SINGLE most specific and accurate product category.

**STRICT REQUIREMENTS:**
1. Select EXACTLY ONE category path per review, OR select NONE if no category fits well
2. Choose the MOST SPECIFIC category that accurately describes the product
3. Be CONSERVATIVE: if uncertain, select NONE (output -1)
4. The category must genuinely match the product in the review

**Instructions:**
- Read each review carefully
- Look at the full category paths provided (from general to specific)
- Select the ONE path that best matches the product
- If none of the paths accurately describe the product, select -1 (NONE)

Respond in JSON format with EXACTLY this structure:
{
    "selections": [
        {"doc_id": <id>, "selected_class_id": <single_id_or_-1>, "reasoning": "brief explanation"},
        ...
    ]
}

**CRITICAL: 
- "selected_class_id" must be a SINGLE integer (the leaf class ID) or -1 for NONE
- Do NOT return an array, return a single integer
- Ancestor classes will be automatically included based on your selection**

Reviews and Category Paths:
"""
    
    for i, (doc_text, candidates) in enumerate(zip(doc_texts, candidates_list)):
        # Truncate review
        truncated_text = doc_text[:800] + ("..." if len(doc_text) > 800 else "")
        
        prompt += f"""
---
Document {i+1}:
Review: {truncated_text}

Category Paths (select ONE most accurate, or -1 if none fit):
"""
        for cid, cname, score in candidates:
            # Build full hierarchy path
            path = build_hierarchy_path(cid, _child_to_parent, _id2class)
            level = get_hierarchy_level(cid, _child_to_parent)
            prompt += f"  - [ID {cid}] (Level {level}): {path}\n"
    
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
        
        # Map results back to doc_ids (single ID only, expansion done in main.py)
        selections = {}
        for i, selection in enumerate(result.get("selections", [])):
            if i < len(doc_ids):
                selected_id = selection.get("selected_class_id", -1)
                
                # Handle case where LLM returns array instead of single int
                if isinstance(selected_id, list):
                    selected_id = selected_id[0] if selected_id else -1
                
                if selected_id == -1 or selected_id is None:
                    # No selection - empty list
                    selections[doc_ids[i]] = []
                else:
                    # Return only the single selected ID (ancestors added in main.py)
                    selections[doc_ids[i]] = [selected_id]
        
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
    edges: List[Tuple[int, int]],  # NEW: hierarchy edges
    ambiguous_doc_ids: List[int],
    max_api_calls: int = MAX_API_CALLS,
    batch_size: int = BATCH_SIZE,
    max_parallel: int = MAX_PARALLEL_CALLS,
    checkpoint_path: str = "checkpoints/llm_refinement_checkpoint.json"
):
    """
    Main function: Use LLM to select ONE most specific category with hierarchy path
    
    Args:
        edges: List of (parent_id, child_id) tuples for hierarchy
        ambiguous_doc_ids: Docs needing LLM judgment (ratio <= 2 cases)
        max_parallel: Maximum number of parallel API calls
    
    Returns selected class + all ancestor classes automatically.
    """
    
    # Initialize hierarchy data for prompts
    set_hierarchy_data(edges, id2class)
    
    print("=" * 100)
    print("LLM-based Pseudo Labeling with Hierarchy Paths")
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
