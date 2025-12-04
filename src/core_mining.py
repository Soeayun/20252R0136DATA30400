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
    Optimized Full NLI Top-down Core Class Mining with Document Batching.
    Processes multiple documents simultaneously to maximize GPU throughput.
    """
    print(f"\n=== Starting Full NLI Top-down Core Class Mining (Batched) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    # Auto-detect entailment index
    entailment_idx = -1
    for k, v in model.config.label2id.items():
        if k.lower().startswith('entail'):
            entailment_idx = v
            break
    if entailment_idx == -1:
        if "deberta" in model_name: entailment_idx = 1
        else: entailment_idx = 2
    
    # Identify Roots
    all_classes = set(id2class.keys())
    all_children = set()
    for children in children_dict.values():
        all_children.update(children)
    roots = list(all_classes - all_children)

    core_classes = {}
    
    # Internal batch size for NLI inference (to avoid OOM)
    # This is different from the document batch_size passed as argument
    INFERENCE_BATCH_SIZE = 512

    def run_nli(premises, hypotheses):
        if not premises: return np.array([])
        scores = []
        for i in range(0, len(premises), INFERENCE_BATCH_SIZE):
            batch_p = premises[i:i+INFERENCE_BATCH_SIZE]
            batch_h = hypotheses[i:i+INFERENCE_BATCH_SIZE]
            inputs = tokenizer(batch_p, batch_h, return_tensors='pt', padding=True, truncation=True, max_length=192).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    batch_scores = probs[:, entailment_idx].float().cpu().numpy()
                    scores.extend(batch_scores)
        return np.array(scores)

    def get_hypothesis(cid):
        cname = id2class[cid]
        base = f"This product is {cname}."
        if class2keywords and cname in class2keywords:
            keywords = class2keywords[cname][:10]
            if keywords:
                base += f" Keywords: {', '.join(keywords)}."
        return base

    # Pre-compute Root Hypotheses
    root_hypotheses_map = {r: get_hypothesis(r) for r in roots}

    # Process documents in batches
    # The 'batch_size' argument is used as the number of documents to process at once
    doc_batch_size = batch_size
    
    for i in tqdm(range(0, len(doc_ids), doc_batch_size), desc="Full NLI Search (Batched)"):
        batch_doc_ids = doc_ids[i : i + doc_batch_size]
        batch_docs_text = [corpus[did] for did in batch_doc_ids]
        
        # State tracking per document in batch
        batch_path_scores = [{} for _ in batch_doc_ids]
        batch_final_candidates = [{} for _ in batch_doc_ids]
        
        # --- Level 0 (Roots) ---
        l0_premises = []
        l0_hypotheses = []
        l0_map = [] # (doc_idx_in_batch, root_id)
        
        for d_idx, doc_text in enumerate(batch_docs_text):
            for r in roots:
                l0_premises.append(doc_text)
                l0_hypotheses.append(root_hypotheses_map[r])
                l0_map.append((d_idx, r))
                
        l0_scores = run_nli(l0_premises, l0_hypotheses)
        
        # Organize scores
        temp_scores = [[] for _ in batch_doc_ids]
        for idx, score in enumerate(l0_scores):
            d_idx, r_id = l0_map[idx]
            temp_scores[d_idx].append((r_id, float(score)))
            
        # Select Top-2 Roots
        batch_selected_l0 = []
        for d_idx in range(len(batch_doc_ids)):
            scores = temp_scores[d_idx]
            scores.sort(key=lambda x: x[1], reverse=True)
            top2 = scores[:2]
            
            selected_ids = []
            for r_id, s in top2:
                batch_path_scores[d_idx][r_id] = s
                batch_final_candidates[d_idx][r_id] = s
                selected_ids.append(r_id)
            batch_selected_l0.append(selected_ids)
            
        # --- Level 1 ---
        l1_premises = []
        l1_hypotheses = []
        l1_map = [] # (doc_idx_in_batch, child_id)
        
        for d_idx, selected_roots in enumerate(batch_selected_l0):
            doc_text = batch_docs_text[d_idx]
            candidates = set()
            for r in selected_roots:
                candidates.update(children_dict.get(r, []))
            
            for c in candidates:
                l1_premises.append(doc_text)
                l1_hypotheses.append(get_hypothesis(c))
                l1_map.append((d_idx, c))
        
        if l1_premises:
            l1_scores = run_nli(l1_premises, l1_hypotheses)
            
            temp_scores = [[] for _ in batch_doc_ids]
            for idx, score in enumerate(l1_scores):
                d_idx, c_id = l1_map[idx]
                
                # Calculate Path Score = Max(Parent_PS * Local)
                max_ps = -1.0
                for p in batch_selected_l0[d_idx]:
                    if c_id in children_dict.get(p, []):
                        p_ps = batch_path_scores[d_idx][p]
                        current_ps = p_ps * float(score)
                        if current_ps > max_ps:
                            max_ps = current_ps
                
                if max_ps > 0:
                    temp_scores[d_idx].append((c_id, max_ps, float(score)))
            
            batch_selected_l1 = []
            for d_idx in range(len(batch_doc_ids)):
                scores = temp_scores[d_idx]
                scores.sort(key=lambda x: x[1], reverse=True)
                top4 = scores[:4]
                
                selected_ids = []
                for c_id, ps, local_s in top4:
                    batch_path_scores[d_idx][c_id] = ps
                    batch_final_candidates[d_idx][c_id] = local_s
                    selected_ids.append(c_id)
                batch_selected_l1.append(selected_ids)
        else:
            batch_selected_l1 = [[] for _ in batch_doc_ids]

        # --- Level 2 ---
        l2_premises = []
        l2_hypotheses = []
        l2_map = []
        
        for d_idx, selected_l1 in enumerate(batch_selected_l1):
            doc_text = batch_docs_text[d_idx]
            candidates = set()
            for p in selected_l1:
                candidates.update(children_dict.get(p, []))
                
            for c in candidates:
                l2_premises.append(doc_text)
                l2_hypotheses.append(get_hypothesis(c))
                l2_map.append((d_idx, c))
                
        if l2_premises:
            l2_scores = run_nli(l2_premises, l2_hypotheses)
            
            temp_scores = [[] for _ in batch_doc_ids]
            for idx, score in enumerate(l2_scores):
                d_idx, c_id = l2_map[idx]
                
                max_ps = -1.0
                for p in batch_selected_l1[d_idx]:
                    if c_id in children_dict.get(p, []):
                        p_ps = batch_path_scores[d_idx][p]
                        current_ps = p_ps * float(score)
                        if current_ps > max_ps:
                            max_ps = current_ps
                            
                if max_ps > 0:
                    temp_scores[d_idx].append((c_id, max_ps, float(score)))
            
            batch_selected_l2 = []
            for d_idx in range(len(batch_doc_ids)):
                scores = temp_scores[d_idx]
                scores.sort(key=lambda x: x[1], reverse=True)
                top9 = scores[:9]
                
                selected_ids = []
                for c_id, ps, local_s in top9:
                    batch_path_scores[d_idx][c_id] = ps
                    batch_final_candidates[d_idx][c_id] = local_s
                    selected_ids.append(c_id)
                batch_selected_l2.append(selected_ids)

        # Store results
        for d_idx, did in enumerate(batch_doc_ids):
            core_classes[did] = batch_final_candidates[d_idx]
            
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
            
    # 3. Filter and Top-3 Selection
    final_core_classes = {} # {doc_id: [core_class1, core_class2]}
    
    for doc_id, confs in doc_confidences.items():
        # Collect candidates that pass the threshold
        valid_candidates = []
        for c, conf in confs.items():
            tau = class_thresholds.get(c, 0.0)
            if conf > tau:
                valid_candidates.append((c, conf))
        
        # Sort by Confidence Score (Descending)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep Top-3
        top_k = valid_candidates[:15]
        
        final_core_classes[doc_id] = [c for c, score in top_k]
        
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
                # masks[i, d] = 0.0
                pass
                
    return targets, masks
