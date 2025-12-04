import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import src.utils as utils
from collections import defaultdict


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
        # [Fix] Retrieve candidates for the current document
        candidates = doc_candidates[doc_id]
        
        # Collect candidates that pass the threshold
        valid_candidates = []
        for c, conf in confs.items():
            tau = class_thresholds.get(c, 0.0)
            if conf > tau:
                valid_candidates.append((c, conf))
        
        # Sort by Confidence Score (Descending)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Keep Top-K
        # [Added] Hard Filter: Raw NLI Score > 0.33
        # Even if confidence is high relative to family, absolute score must be reasonable.
        final_candidates = []
        for c, conf in valid_candidates:
            raw_score = candidates.get(c, 0.0)
            if raw_score > 0.33:
                final_candidates.append(c)
            
            if len(final_candidates) >= 15:
                break
        
        final_core_classes[doc_id] = final_candidates
        
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
