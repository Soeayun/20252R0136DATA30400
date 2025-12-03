import torch
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import src.utils as utils
from collections import defaultdict


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
    Updated with Path Score (ps) and Strict Candidate Selection.
    """
    print(f"Loading Entailment Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    # Auto-detect entailment index
    entailment_idx = model.config.label2id.get('ENTAILED', model.config.label2id.get('entailment', -1))
    if entailment_idx == -1:
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

    doc_candidates = {} 

    # Build BM25 index
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
    
    def run_nli(premises, hypotheses):
        all_scores = []
        for i in range(0, len(premises), batch_size):
            batch_p = premises[i:i+batch_size]
            batch_h = hypotheses[i:i+batch_size]
            if not batch_p: continue
            inputs = tokenizer(batch_p, batch_h, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                scores = probs[:, entailment_idx].cpu().numpy()
                all_scores.extend(scores)
        return all_scores

    for doc_id in tqdm(doc_ids, desc="Hybrid Top-down Search"):
        doc_text = corpus[doc_id]
        doc_tokens = doc_text.lower().split()
        
        # Store ONLY selected candidates (Top-K)
        candidates_dict = {} 
        
        # --- Level 0 (Roots) ---
        root_premises = [doc_text] * len(roots)
        root_hypotheses = [f"This example is {id2class[r]}." for r in roots]
        
        root_sim_scores = run_nli(root_premises, root_hypotheses)
        
        # Path Score (L0): ps = sim
        l0_scored = []
        for r, s in zip(roots, root_sim_scores):
            l0_scored.append((r, float(s)))
            
        # Select Top-2
        l0_scored.sort(key=lambda x: x[1], reverse=True)
        selected_l0 = l0_scored[:2]
        
        for c, s in selected_l0:
            candidates_dict[c] = s
            
        # --- Level 1 ---
        l1_candidates_info = [] # (child_id, parent_ps)
        for p_id, p_ps in selected_l0:
            kids = children_dict.get(p_id, [])
            for k in kids:
                l1_candidates_info.append((k, p_ps))
        
        if not l1_candidates_info:
            doc_candidates[doc_id] = candidates_dict
            continue
            
        l1_ids = [x[0] for x in l1_candidates_info]
        l1_premises = [doc_text] * len(l1_ids)
        l1_hypotheses = [f"This example is {id2class[c]}." for c in l1_ids]
        
        l1_sim_scores = run_nli(l1_premises, l1_hypotheses)
        
        # Path Score (L1): ps = parent_ps * sim
        l1_scored = []
        for i, (c_id, p_ps) in enumerate(l1_candidates_info):
            sim = float(l1_sim_scores[i])
            ps = p_ps * sim
            l1_scored.append((c_id, ps))
            
        # Select Top-4
        l1_scored.sort(key=lambda x: x[1], reverse=True)
        selected_l1 = l1_scored[:4]
        
        for c, s in selected_l1:
            candidates_dict[c] = s
            
        # --- Level 2 ---
        l2_candidates_info = [] # (child_id, parent_ps)
        for p_id, p_ps in selected_l1:
            kids = children_dict.get(p_id, [])
            for k in kids:
                l2_candidates_info.append((k, p_ps))
                
        if not l2_candidates_info:
            doc_candidates[doc_id] = candidates_dict
            continue
            
        # BM25 Filter for L2
        all_bm25_scores = bm25.get_scores(doc_tokens)
        
        l2_bm25_scored = []
        for c_id, p_ps in l2_candidates_info:
            idx = cid_to_idx[c_id]
            bm25_score = all_bm25_scores[idx]
            l2_bm25_scored.append((c_id, p_ps, bm25_score))
            
        # Sort by BM25 and take Top-4
        l2_bm25_scored.sort(key=lambda x: x[2], reverse=True)
        l2_top4 = l2_bm25_scored[:4]
        
        # NLI on Top-4
        l2_ids = [x[0] for x in l2_top4]
        l2_premises = [doc_text] * len(l2_ids)
        l2_hypotheses = [f"This example is {id2class[c]}." for c in l2_ids]
        
        l2_sim_scores = run_nli(l2_premises, l2_hypotheses)
        
        # Path Score (L2): ps = parent_ps * sim
        l2_scored = []
        for i, (c_id, p_ps, _) in enumerate(l2_top4):
            sim = float(l2_sim_scores[i])
            ps = p_ps * sim
            l2_scored.append((c_id, ps))
            
        # Select Top-3 (Final)
        l2_scored.sort(key=lambda x: x[1], reverse=True)
        selected_l2 = l2_scored[:3]
        
        for c, s in selected_l2:
            candidates_dict[c] = s
        
        doc_candidates[doc_id] = candidates_dict

    return doc_candidates

def generate_core_classes_full_nli(corpus, id2class, doc_ids, parents_dict, children_dict, device, 
                                   model_name="cross-encoder/nli-deberta-v3-base", 
                                   batch_size=32, 
                                   class2keywords=None):
    """
    Full NLI Top-down Core Class Mining (TaxoClass Original).
    No BM25 filtering. Evaluates NLI on ALL children at each step.
    
    Strategy:
    - Level 0: NLI on all Roots -> Top-2
    - Level 1: NLI on ALL children of selected Roots -> Top-4
    - Level 2: NLI on ALL children of selected Level 1 -> Top-9
    """
    print(f"\n=== Starting Full NLI Top-down Core Class Mining (No BM25) ===")
    
    # Load NLI Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    entailment_idx = -1
    for k, v in model.config.label2id.items():
        if k.lower().startswith('entail'):
            entailment_idx = v
            break
    if entailment_idx == -1:
        # Fallback for DeBERTa-v3-base if not found in config
        if "deberta" in model_name:
            entailment_idx = 1
        else:
            entailment_idx = 2
    print(f"Entailment Index: {entailment_idx}")

    # 1. Identify Roots
    all_classes = set(id2class.keys())
    all_children = set()
    for children in children_dict.values():
        all_children.update(children)
    roots = list(all_classes - all_children)
    print(f"Identified {len(roots)} Root classes.")
    
    core_classes = {}
    
    # Helper for NLI
    def run_nli(premises, hypotheses):
        scores = []
        for i in range(0, len(premises), batch_size):
            batch_p = premises[i:i+batch_size]
            batch_h = hypotheses[i:i+batch_size]
            inputs = tokenizer(batch_p, batch_h, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                batch_scores = probs[:, entailment_idx].cpu().numpy()
                scores.extend(batch_scores)
        return np.array(scores)

    for doc_id in tqdm(doc_ids, desc="Full NLI Search"):
        doc_text = corpus[doc_id]
        
        # --- Level 0 (Roots) ---
        # Candidates: All Roots
        l0_candidates = roots
        
        # NLI on All Roots
        l0_premises = [doc_text] * len(l0_candidates)
        l0_hypotheses = [f"This example is {id2class[c]}." for c in l0_candidates]
        
        l0_scores = run_nli(l0_premises, l0_hypotheses)
        
        # Select Top-2 Roots
        top_k_l0 = 2
        if len(l0_candidates) > top_k_l0:
            top_indices = np.argsort(l0_scores)[-top_k_l0:]
            selected_l0 = [l0_candidates[i] for i in top_indices]
        else:
            selected_l0 = l0_candidates
            
        # --- Level 1 ---
        l1_candidates = []
        for p in selected_l0:
            l1_candidates.extend(children_dict.get(p, []))
        l1_candidates = list(set(l1_candidates))
        
        if not l1_candidates:
            core_classes[doc_id] = selected_l0
            continue
            
        # NLI on ALL Level 1 Candidates (No BM25)
        l1_premises = [doc_text] * len(l1_candidates)
        l1_hypotheses = [f"This example is {id2class[c]}." for c in l1_candidates]
        
        l1_scores = run_nli(l1_premises, l1_hypotheses)
        
        # Select Top-4 Level 1
        top_k_l1 = 4
        if len(l1_candidates) > top_k_l1:
            top_indices = np.argsort(l1_scores)[-top_k_l1:]
            selected_l1 = [l1_candidates[i] for i in top_indices]
        else:
            selected_l1 = l1_candidates
            
        # --- Level 2 ---
        l2_candidates = []
        for p in selected_l1:
            l2_candidates.extend(children_dict.get(p, []))
        l2_candidates = list(set(l2_candidates))
        
        if not l2_candidates:
            core_classes[doc_id] = selected_l1
            continue
            
        # NLI on ALL Level 2 Candidates (No BM25)
        l2_premises = [doc_text] * len(l2_candidates)
        l2_hypotheses = [f"This example is {id2class[c]}." for c in l2_candidates]
        
        l2_scores = run_nli(l2_premises, l2_hypotheses)
        
        # Select Top-9 Level 2
        top_k_l2 = 9
        if len(l2_candidates) > top_k_l2:
            top_indices = np.argsort(l2_scores)[-top_k_l2:]
            selected_l2 = [l2_candidates[i] for i in top_indices]
        else:
            selected_l2 = l2_candidates
            
        # --- Level 3 (Handle ID 44 and potential others) ---
        l3_candidates = []
        for p in selected_l2:
            l3_candidates.extend(children_dict.get(p, []))
        l3_candidates = list(set(l3_candidates))
        
        if l3_candidates:
            # NLI on ALL Level 3 Candidates
            l3_premises = [doc_text] * len(l3_candidates)
            l3_hypotheses = [f"This example is {id2class[c]}." for c in l3_candidates]
            
            l3_scores = run_nli(l3_premises, l3_hypotheses)
            
            # Select Top-3 Level 3 (Arbitrary small number, usually 7 total)
            # Since there are only 7, we can just keep them all or Top-3
            top_k_l3 = 3
            if len(l3_candidates) > top_k_l3:
                top_indices = np.argsort(l3_scores)[-top_k_l3:]
                selected_l3 = [l3_candidates[i] for i in top_indices]
            else:
                selected_l3 = l3_candidates
        else:
            selected_l3 = []
            l3_scores = []
            
        # Final Candidates: Level 2 + Level 3 selections
        # Note: We need to return a dict of {class_id: score} for identify_confident_core_classes
        
        candidates_dict = {}
        
        # Add L0 scores
        for c, s in zip(l0_candidates, l0_scores):
            candidates_dict[c] = float(s)
            
        # Add L1 scores
        for c, s in zip(l1_candidates, l1_scores):
            candidates_dict[c] = float(s)
            
        # Add L2 scores
        for c, s in zip(l2_candidates, l2_scores):
            candidates_dict[c] = float(s)
            
        # Add L3 scores
        for c, s in zip(l3_candidates, l3_scores):
            candidates_dict[c] = float(s)
            
        core_classes[doc_id] = candidates_dict
        
    return core_classes

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
