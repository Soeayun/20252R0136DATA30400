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
    FIXED Full NLI Top-down Core Class Mining.
    1. Calculates Path Score (Parent * Local).
    2. Stores ONLY selected candidates in the final dictionary.
    """
    print(f"\n=== Starting Full NLI Top-down Core Class Mining (Fixed) ===")
    
    # ... (모델 로딩 부분은 동일) ...
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    # ... (Entailment Index 찾기 동일) ...
    entailment_idx = -1
    for k, v in model.config.label2id.items():
        if k.lower().startswith('entail'):
            entailment_idx = v
            break
    if entailment_idx == -1:
        if "deberta" in model_name: entailment_idx = 1
        else: entailment_idx = 2
    
    # ... (Roots 식별 동일) ...
    all_classes = set(id2class.keys())
    all_children = set()
    for children in children_dict.values():
        all_children.update(children)
    roots = list(all_classes - all_children)

    core_classes = {}
    
    # Helper for NLI (동일)
    def run_nli(premises, hypotheses):
        if not premises: return np.array([])
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
        
        # 결과를 저장할 dict (Local Score 저장용 -> Confidence 계산에 사용)
        candidates_dict = {} 
        # 랭킹을 위한 Path Score 추적용
        path_scores_map = {}

        # --- Level 0 (Roots) ---
        l0_candidates = roots
        l0_premises = [doc_text] * len(l0_candidates)
        l0_hypotheses = [f"This example is {id2class[c]}." for c in l0_candidates]
        l0_local_scores = run_nli(l0_premises, l0_hypotheses)
        
        # [수정] Root의 Path Score = Local Score
        for c, s in zip(l0_candidates, l0_local_scores):
            path_scores_map[c] = float(s)
        
        # [수정] Path Score 기준으로 Top-2 선정
        top_k_l0 = 2
        l0_sorted = sorted(l0_candidates, key=lambda c: path_scores_map[c], reverse=True)
        selected_l0 = l0_sorted[:top_k_l0]

        # [수정] 선택된 Root만 저장
        l0_local_map = dict(zip(l0_candidates, l0_local_scores))
        for c in selected_l0:
            candidates_dict[c] = float(l0_local_map[c])

        # --- Level 1 ---
        l1_candidates = []
        l1_parents_map = {} # 부모 추적용
        for p in selected_l0:
            for child in children_dict.get(p, []):
                l1_candidates.append(child)
                l1_parents_map[child] = p
        l1_candidates = list(set(l1_candidates))
        
        if not l1_candidates:
            core_classes[doc_id] = candidates_dict
            continue
            
        l1_premises = [doc_text] * len(l1_candidates)
        l1_hypotheses = [f"This example is {id2class[c]}." for c in l1_candidates]
        l1_local_scores = run_nli(l1_premises, l1_hypotheses)
        
        # [수정] Path Score 계산: Parent_PS * Local_Score
        l1_local_map = dict(zip(l1_candidates, l1_local_scores))
        for c in l1_candidates:
            parent = l1_parents_map[c]
            parent_ps = path_scores_map[parent]
            local_s = float(l1_local_map[c])
            path_scores_map[c] = parent_ps * local_s # 누적 곱
            
        # [수정] Path Score 기준으로 Top-4 선정
        top_k_l1 = 4
        l1_sorted = sorted(l1_candidates, key=lambda c: path_scores_map[c], reverse=True)
        selected_l1 = l1_sorted[:top_k_l1]
        
        # [수정] 선택된 L1만 저장
        for c in selected_l1:
            candidates_dict[c] = float(l1_local_map[c])

        # --- Level 2 ---
        l2_candidates = []
        l2_parents_map = {}
        for p in selected_l1:
            for child in children_dict.get(p, []):
                l2_candidates.append(child)
                l2_parents_map[child] = p
        l2_candidates = list(set(l2_candidates))
        
        if not l2_candidates:
            core_classes[doc_id] = candidates_dict
            continue
            
        l2_premises = [doc_text] * len(l2_candidates)
        l2_hypotheses = [f"This example is {id2class[c]}." for c in l2_candidates]
        l2_local_scores = run_nli(l2_premises, l2_hypotheses)
        
        # [수정] Path Score 계산
        l2_local_map = dict(zip(l2_candidates, l2_local_scores))
        for c in l2_candidates:
            parent = l2_parents_map[c]
            parent_ps = path_scores_map[parent]
            local_s = float(l2_local_map[c])
            path_scores_map[c] = parent_ps * local_s
            
        # [수정] Path Score 기준으로 Top-9 선정
        top_k_l2 = 9
        l2_sorted = sorted(l2_candidates, key=lambda c: path_scores_map[c], reverse=True)
        selected_l2 = l2_sorted[:top_k_l2]

        # [수정] 선택된 L2만 저장
        for c in selected_l2:
            candidates_dict[c] = float(l2_local_map[c])

        # --- Level 3 (Optional) ---
        l3_candidates = []
        l3_parents_map = {}
        for p in selected_l2:
            for child in children_dict.get(p, []):
                l3_candidates.append(child)
                l3_parents_map[child] = p
        l3_candidates = list(set(l3_candidates))

        if l3_candidates:
            l3_premises = [doc_text] * len(l3_candidates)
            l3_hypotheses = [f"This example is {id2class[c]}." for c in l3_candidates]
            l3_local_scores = run_nli(l3_premises, l3_hypotheses)
            
            l3_local_map = dict(zip(l3_candidates, l3_local_scores))
            for c in l3_candidates:
                parent = l3_parents_map[c]
                parent_ps = path_scores_map[parent]
                local_s = float(l3_local_map[c])
                path_scores_map[c] = parent_ps * local_s
            
            # Top-3 (임의 설정)
            top_k_l3 = 3
            l3_sorted = sorted(l3_candidates, key=lambda c: path_scores_map[c], reverse=True)
            selected_l3 = l3_sorted[:top_k_l3]
            
            for c in selected_l3:
                candidates_dict[c] = float(l3_local_map[c])

        core_classes[doc_id] = candidates_dict
        
    return core_classes

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
