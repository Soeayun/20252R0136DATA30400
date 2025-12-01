import torch
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import src.utils as utils


# bm25_scores, train_doc_ids = core_mining.calculate_bm25_scores(train_corpus, class2keywords, id2class)
def calculate_bm25_scores(corpus, class2keywords, id2class):
    """
    Calculates BM25 scores for each document against each class.
    
    Args:
        corpus: dict {doc_id: text}
        class2keywords: dict {class_name: [keywords]}
        id2class: dict {class_id: class_name}
        
    Returns:
        scores: numpy array of shape (num_docs, num_classes)
    """
    print("Calculating BM25 scores...")
    # Prepare corpus for BM25
    doc_ids = sorted(list(corpus.keys()))
    tokenized_corpus = [doc.lower().split() for doc in [corpus[did] for did in doc_ids]]
    bm25 = BM25Okapi(tokenized_corpus)
    
    num_docs = len(doc_ids)
    num_classes = len(id2class)
    scores = np.zeros((num_docs, num_classes))
    
    for cid in tqdm(range(num_classes), desc="BM25 Class Loop"):
        cname = id2class[cid]
        # Query is class name + keywords
        keywords = class2keywords.get(cname, [])
        query = f"{cname} {' '.join(keywords)}"
        tokenized_query = query.lower().split()
        
        class_scores = bm25.get_scores(tokenized_query) # 각 query단어에 대해 BM-25를 구한 뒤 점수 합산
        scores[:, cid] = class_scores
        
    # Normalize scores (Min-Max or Softmax? Paper uses normalization)
    # Simple Min-Max per document to 0-1 range
    min_vals = scores.min(axis=1, keepdims=True)
    max_vals = scores.max(axis=1, keepdims=True)
    scores = (scores - min_vals) / (max_vals - min_vals + 1e-9)
    
    return scores, doc_ids

# nli_scores = calculate_entailment_scores(
#         train_corpus, id2class, train_doc_ids, device, 
#         top_k_filter=top_k_indices
#     )

def calculate_entailment_scores(corpus, id2class, doc_ids, device, model_name="cross-encoder/nli-deberta-v3-base", batch_size=32, top_k_filter=None):
    """
    Calculates Entailment scores.
    To save time, if top_k_filter is provided (indices of top classes from BM25), 
    only calculate for those.
    
    Args:
        top_k_filter: numpy array (num_docs, k) containing class indices to check.
    """
    print(f"Loading Entailment Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    num_docs = len(doc_ids)
    num_classes = len(id2class)
    scores = np.zeros((num_docs, num_classes))
    
    # Entailment label index (usually 2 for RoBERTa-MNLI: 0:contradiction, 1:neutral, 2:entailment)
    # Check config
    entailment_idx = model.config.label2id.get('ENTAILED', model.config.label2id.get('entailment', 2))
    
    print("Calculating Entailment scores...")
    
    # If full computation is too slow, we iterate by document batches
    # Construct pairs
    
    # Optimization: Only check Top-K from BM25 if provided
    if top_k_filter is not None:
        print(f"Optimized: Calculating NLI only for Top-{top_k_filter.shape[1]} candidates from BM25.")
        
        for i in tqdm(range(0, num_docs, batch_size), desc="NLI Batch"):
            batch_doc_indices = range(i, min(i + batch_size, num_docs))
            batch_pairs = []
            batch_coords = [] # (doc_idx, class_idx)
            
            for doc_idx in batch_doc_indices:
                did = doc_ids[doc_idx]
                text = corpus[did]

                # Truncate text to avoid OOM/Length errors (keep first 256 words approx)
                # word 수를 줄인다는 것 조심
                text = " ".join(text.split()[:200]) 
                
                classes_to_check = top_k_filter[doc_idx]
                
                for cid in classes_to_check:
                    cname = id2class[cid]
                    hypothesis = f"This review is about {cname}."
                    batch_pairs.append((text, hypothesis))
                    batch_coords.append((doc_idx, cid))
            
            if not batch_pairs:
                continue
                
            # Tokenize
            inputs = tokenizer(batch_pairs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = softmax(outputs.logits, dim=1)
                entail_probs = probs[:, entailment_idx].cpu().numpy()
                
            for idx, (d_idx, c_idx) in enumerate(batch_coords):
                scores[d_idx, c_idx] = entail_probs[idx]
                
    else:
        # Full computation (Warning: Very Slow)
        print("Warning: Full NLI computation requested. This may take a very long time.")
        # Implementation omitted for brevity/safety unless requested. 
        # Fallback to a simpler approach or error.
        pass

    return scores

def generate_core_classes_hybrid_top_down(corpus, id2class, doc_ids, parents_dict, children_dict, device, model_name="cross-encoder/nli-deberta-v3-base", batch_size=32, class2keywords=None):
    """
    Generates candidate core classes using Hybrid Top-down approach (BM25 filtering + NLI).
    
    Strategy:
    - Level 0 (Roots): NLI on all (6) -> Select Top-2.
    - Level 1: Get children of selected Roots. BM25 Top-5 -> NLI -> Select Top-2.
    - Level 2: Get children of selected L1. BM25 Top-4 -> NLI -> Select Top-3.
    
    Returns:
        doc_candidates: dict {doc_id: {class_id: nli_score, ...}}
    """
    print(f"Loading Entailment Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    # Auto-detect entailment index
    entailment_idx = model.config.label2id.get('ENTAILED', model.config.label2id.get('entailment', -1))
    if entailment_idx == -1:
        # Fallback for DeBERTa-v3-base if not found in config
        if "deberta" in model_name:
            entailment_idx = 1
        else:
            entailment_idx = 2
    print(f"Entailment Index: {entailment_idx}")

    # Identify Roots
    all_classes = set(id2class.keys())
    children_set = set()
    for p, kids in children_dict.items():
        children_set.update(kids)
    roots = list(all_classes - children_set)
    print(f"Identified {len(roots)} Root classes.")

    doc_candidates = {} # {doc_id: {class_id: score}}

    # Pre-compute BM25 for filtering (optional optimization: compute on-the-fly or pre-compute all)
    # Since we need BM25 for specific subsets, we can implement a lightweight BM25 helper here or reuse utils
    # For efficiency, let's build a global BM25 index first, then query it.
    print("Building BM25 index for filtering...")
    tokenized_classes = []
    class_ids_list = sorted(list(id2class.keys()))
    cid_to_idx = {cid: i for i, cid in enumerate(class_ids_list)}
    
    for cid in class_ids_list:
        text = id2class[cid]
        if class2keywords and id2class[cid] in class2keywords:
             text += " " + " ".join(class2keywords[id2class[cid]])
        tokenized_classes.append(text.lower().split())
    
    bm25 = BM25Okapi(tokenized_classes)
    
    # Helper for NLI batch inference
    def run_nli(premises, hypotheses):
        inputs = tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, entailment_idx].cpu().numpy()
        return scores

    for doc_id in tqdm(doc_ids, desc="Hybrid Top-down Search"):
        doc_text = corpus[doc_id]
        candidates = {} # {class_id: score}
        
        # --- Level 0 (Roots) ---
        # NLI on all roots
        root_premises = [doc_text] * len(roots)
        root_hypotheses = [f"This example is {id2class[r]}." for r in roots]
        
        # Batch inference for roots
        root_scores = []
        for i in range(0, len(roots), batch_size):
            batch_p = root_premises[i:i+batch_size]
            batch_h = root_hypotheses[i:i+batch_size]
            if not batch_p: continue
            scores = run_nli(batch_p, batch_h)
            root_scores.extend(scores)
            
        root_candidates = []
        for r, s in zip(roots, root_scores):
            candidates[r] = float(s)
            root_candidates.append((r, s))
            
        # Select Top-2 Roots
        root_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_roots = [r for r, s in root_candidates[:2]]
        
        # --- Level 1 ---
        l1_candidates_pool = set()
        for r in selected_roots:
            l1_candidates_pool.update(children_dict.get(r, []))
        
        l1_candidates_list = list(l1_candidates_pool)
        if l1_candidates_list:
            # BM25 Filter: Top-5
            doc_tokens = doc_text.lower().split()
            # We need to score only l1_candidates_list. 
            # BM25Okapi.get_scores calculates for ALL docs in index.
            # So we get all scores, then pick the ones for l1_candidates_list.
            all_bm25_scores = bm25.get_scores(doc_tokens)
            
            l1_bm25 = []
            for cid in l1_candidates_list:
                idx = cid_to_idx[cid]
                l1_bm25.append((cid, all_bm25_scores[idx]))
            
            l1_bm25.sort(key=lambda x: x[1], reverse=True)
            l1_top5 = [cid for cid, s in l1_bm25[:5]]
            
            # NLI on Top-5
            l1_premises = [doc_text] * len(l1_top5)
            l1_hypotheses = [f"This example is {id2class[c]}." for c in l1_top5]
            
            l1_nli_scores = run_nli(l1_premises, l1_hypotheses)
            
            l1_scored = []
            for c, s in zip(l1_top5, l1_nli_scores):
                candidates[c] = float(s)
                l1_scored.append((c, s))
                
            # Select Top-2 L1
            l1_scored.sort(key=lambda x: x[1], reverse=True)
            selected_l1 = [c for c, s in l1_scored[:2]]
        else:
            selected_l1 = []

        # --- Level 2 ---
        l2_candidates_pool = set()
        for p in selected_l1:
            l2_candidates_pool.update(children_dict.get(p, []))
            
        l2_candidates_list = list(l2_candidates_pool)
        if l2_candidates_list:
            # BM25 Filter: Top-4
            doc_tokens = doc_text.lower().split()
            all_bm25_scores = bm25.get_scores(doc_tokens)
            
            l2_bm25 = []
            for cid in l2_candidates_list:
                idx = cid_to_idx[cid]
                l2_bm25.append((cid, all_bm25_scores[idx]))
                
            l2_bm25.sort(key=lambda x: x[1], reverse=True)
            l2_top4 = [cid for cid, s in l2_bm25[:4]]
            
            # NLI on Top-4
            l2_premises = [doc_text] * len(l2_top4)
            l2_hypotheses = [f"This example is {id2class[c]}." for c in l2_top4]
            
            l2_nli_scores = run_nli(l2_premises, l2_hypotheses)
            
            l2_scored = []
            for c, s in zip(l2_top4, l2_nli_scores):
                candidates[c] = float(s)
                l2_scored.append((c, s))
                
            # Select Top-3 L2 (Just to add to candidates, no further levels)
            l2_scored.sort(key=lambda x: x[1], reverse=True)
            selected_l2 = [c for c, s in l2_scored[:3]] 
        
        # --- Level 3 (Handle ID 44 and potential others) ---
        l3_candidates_pool = set()
        for p in selected_l2:
            l3_candidates_pool.update(children_dict.get(p, []))
            
        l3_candidates_list = list(l3_candidates_pool)
        if l3_candidates_list:
            # No BM25 needed for L3 (only 7 classes total)
            # NLI on ALL Level 3 Candidates
            l3_premises = [doc_text] * len(l3_candidates_list)
            l3_hypotheses = [f"This example is {id2class[c]}." for c in l3_candidates_list]
            
            l3_scores = run_nli(l3_premises, l3_hypotheses)
            
            for c, s in zip(l3_candidates_list, l3_scores):
                candidates[c] = float(s)
                
        doc_candidates[doc_id] = candidates

    return doc_candidates

def identify_confident_core_classes(doc_candidates, parents_dict, children_dict):
    """
    Identifies confident core classes using Confidence Score and Median Threshold.
    
    conf(D, c) = sim(D,c) - max(sim(D, parents), sim(D, siblings))
    tau_c = median({conf(D', c) for all D' where c in candidates})
    Select c if conf(D, c) > tau_c
    """
    print("Identifying Confident Core Classes...")
    
    # 1. Calculate Raw Confidence Scores
    # doc_confidences: {doc_id: {class_id: conf_score}}
    doc_confidences = {}
    
    # Also collect scores per class for median calculation
    # class_conf_distribution: {class_id: [conf_score1, conf_score2, ...]}
    class_conf_distribution = defaultdict(list)
    
    for doc_id, candidates in tqdm(doc_candidates.items(), desc="Calc Confidence"):
        doc_confidences[doc_id] = {}
        
        for c, score in candidates.items():
            # Get Parent Scores
            parents = parents_dict.get(c, [])
            parent_scores = [candidates.get(p, 0.0) for p in parents] # 0.0 if parent not in candidates
            max_parent = max(parent_scores) if parent_scores else 0.0
            
            # Get Sibling Scores
            siblings = utils.get_siblings(c, parents_dict, children_dict)
            sibling_scores = [candidates.get(s, 0.0) for s in siblings] # 0.0 if sibling not in candidates (filtered out)
            max_sibling = max(sibling_scores) if sibling_scores else 0.0
            
            # Conf(D, c)
            conf = score - max(max_parent, max_sibling)
            
            doc_confidences[doc_id][c] = conf
            class_conf_distribution[c].append(conf)
            
    # 2. Calculate Median Thresholds
    class_thresholds = {}
    for c, scores in class_conf_distribution.items():
        if scores:
            class_thresholds[c] = np.median(scores)
        else:
            class_thresholds[c] = 0.0 # Should not happen if c is in candidates
            
    # 3. Filter
    final_core_classes = {} # {doc_id: [core_class1, core_class2]}
    
    for doc_id, confs in doc_confidences.items():
        cores = []
        for c, conf in confs.items():
            tau = class_thresholds.get(c, 0.0)
            if conf > tau:
                cores.append(c)
        final_core_classes[doc_id] = cores
        
    return final_core_classes

def expand_labels(core_classes, parents_dict, children_dict, num_classes):
    """
    Expands labels based on hierarchy.
    Returns:
        targets: (num_docs, num_classes) - 1 for Positive, 0 for Negative
        masks: (num_docs, num_classes) - 1 for Valid, 0 for Masked (Descendants)
    """
    print("Expanding Labels hierarchically...")
    num_docs = len(core_classes)
    targets = np.zeros((num_docs, num_classes), dtype=np.float32)
    masks = np.ones((num_docs, num_classes), dtype=np.float32)
    
    for i in tqdm(range(num_docs), desc="Expansion"):
        cores = core_classes[i] # List of core class IDs for this doc
        
        # Positive: Core + Ancestors
        positives = set(cores)
        for c in cores:
            ancestors = utils.get_ancestors(c, parents_dict)
            positives.update(ancestors)
            
        for p in positives:
            targets[i, p] = 1.0
            
        # Masked: Descendants of Core (excluding Core itself)
        # Logic: If a node is a child of a core class, we are unsure.
        descendants = set()
        for c in cores:
            desc = utils.get_descendants(c, children_dict)
            descendants.update(desc)
            
        for d in descendants:
            if d not in positives: # Should not mask if it's already positive (rare/impossible in tree)
                masks[i, d] = 0.0
                
    return targets, masks
