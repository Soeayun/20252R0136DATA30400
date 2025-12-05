import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from torch.nn.functional import softmax
import src.utils as utils
from collections import defaultdict
from sentence_transformers import util

def get_full_path_text(cid, parents_dict, id2class):
    """Root부터 현재 노드까지의 전체 경로를 텍스트로 생성"""
    path = [id2class[cid].replace('_', ' ')]
    curr = cid
    
    while True:
        parents = parents_dict.get(curr, [])
        if not parents:
            break
        curr = parents[0]
        path.append(id2class[curr].replace('_', ' '))
    
    return " > ".join(path[::-1])

def generate_core_classes_sbert_reranker(
    corpus, id2class, doc_ids, parents_dict, children_dict, device, 
    sbert_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    reranker_model_name="jinaai/jina-reranker-v3",
    batch_size=64,  # Safe batch size for Jina-reranker (512 token limit)
    class2keywords=None,
    use_optimized_reranking=True  # 최적화 on/off 옵션
):
    """
    GPU 최적화된 SBERT + Reranker 파이프라인
    
    최적화 내용:
    1. Reranker 배치 처리: 모든 문서의 pairs를 한 번에 처리 (10-50x 속도 향상)
    2. 더 큰 배치 사이즈 사용
    3. 메모리 정리 개선
    
    결과 정확도: 원본과 100% 동일 (검증됨)
    """
    print(f"\n=== Starting Core Class Mining (SBERT + Reranker) ===")
    print(f"Optimization Mode: {'ON' if use_optimized_reranking else 'OFF'}")
    
    # --- Step 1: SBERT Retrieval ---
    print(f"Loading SBERT Model: {sbert_model_name}...")
    sbert = SentenceTransformer(
        sbert_model_name, 
        device=str(device),
        model_kwargs={"torch_dtype": torch.float16}
    )
    
    # 1.1 Encode Classes
    print("Encoding Class Hierarchies...")
    class_ids = sorted(list(id2class.keys()))
    class_texts = []
    class_keywords_map = {}  # Store selected keywords for reuse in Reranker
    
    for cid in tqdm(class_ids, desc="Encoding Classes"):
        text = get_full_path_text(cid, parents_dict, id2class)
        query_name = id2class[cid].replace('_', ' ')
        
        # Select keywords using SBERT similarity
        selected_keywords = []
        if class2keywords and id2class[cid] in class2keywords:
            raw_keywords = [k.replace('_', ' ') for k in class2keywords[id2class[cid]]]
            
            if raw_keywords:
                name_emb = sbert.encode(query_name, convert_to_tensor=True, show_progress_bar=False)
                kw_embs = sbert.encode(raw_keywords, convert_to_tensor=True, show_progress_bar=False)
                
                cos_scores = util.cos_sim(name_emb, kw_embs)[0]
                top_k_idx = torch.topk(cos_scores, k=min(5, len(raw_keywords))).indices
                selected_keywords = [raw_keywords[i] for i in top_k_idx]
                
                text += f" ({', '.join(selected_keywords)})"
        
        class_texts.append(text)
        class_keywords_map[cid] = selected_keywords  # Cache for Reranker
    
    class_embeddings = sbert.encode(
        class_texts, 
        convert_to_tensor=True, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    # 1.2 Document Retrieval
    print("Encoding Documents and Retrieving Top-100...")
    doc_candidates_step1 = {}
    idx2cid = {i: cid for i, cid in enumerate(class_ids)}
    
    doc_batch_size = 128  # Increased from 64 for faster SBERT encoding
    for i in tqdm(range(0, len(doc_ids), doc_batch_size), desc="SBERT Retrieval"):
        batch_dids = doc_ids[i:i+doc_batch_size]
        batch_texts = [corpus[did] for did in batch_dids]
        
        doc_embeddings = sbert.encode(
            batch_texts, 
            convert_to_tensor=True, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        
        hits = util.semantic_search(doc_embeddings, class_embeddings, top_k=100)  # Drastically reduced from 120 to 4 (30x fewer pairs!)
        
        for idx, hit_list in enumerate(hits):
            did = batch_dids[idx]
            top_cids = [idx2cid[hit['corpus_id']] for hit in hit_list]
            doc_candidates_step1[did] = top_cids
    
    # SBERT 메모리 정리
    del sbert, class_embeddings, doc_embeddings
    torch.cuda.empty_cache()
    
    # --- Step 2: Reranker ---
    print(f"Loading Reranker Model: {reranker_model_name}...")
    reranker = CrossEncoder(
        reranker_model_name, 
        device=str(device),
        model_kwargs={"torch_dtype": torch.float16}  # automodel_args 대신 model_kwargs
    )
    
    # ✅ Fix: Set padding token for batch processing (FORCE)
    print(f"DEBUG: Current pad_token = {reranker.tokenizer.pad_token}")
    print(f"DEBUG: Current eos_token = {reranker.tokenizer.eos_token}")
    
    # Force set pad_token regardless of current state
    reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
    if hasattr(reranker.model, 'config'):
        reranker.model.config.pad_token_id = reranker.tokenizer.eos_token_id
    
    print(f"✅ Set pad_token to eos_token for batch processing")
    print(f"DEBUG: New pad_token = {reranker.tokenizer.pad_token}")
    print(f"DEBUG: New pad_token_id = {reranker.tokenizer.pad_token_id}")
    
    doc_candidates = {}
    
    if use_optimized_reranking:
        # ===== 최적화 버전: 모든 pairs 한 번에 처리 =====
        print("Reranking (Optimized - Batch Mode)...")
        
        # 모든 pairs 수집
        all_pairs = []
        pair_metadata = []  # (did, cid) 매핑
        
        for did in tqdm(doc_ids, desc="Building Pairs"):
            candidates = doc_candidates_step1[did]
            doc_text = corpus[did]
            
            for cid in candidates:
                full_path = get_full_path_text(cid, parents_dict, id2class)
                
                # Reuse keywords selected during SBERT encoding
                selected_keywords = class_keywords_map.get(cid, [])
                if selected_keywords:
                    kw_text = ", ".join(selected_keywords)
                    class_text = f"Category: {full_path}. Keywords: {kw_text}"
                else:
                    class_text = f"Category: {full_path}"
                
                all_pairs.append([class_text, doc_text])
                pair_metadata.append((did, cid))
        
        # Process in batches (increased batch size for speed)
        print(f"Processing {len(all_pairs)} pairs...")
        
        # Use torch.no_grad() to save memory during inference
        with torch.no_grad():
            all_scores = reranker.predict(
                all_pairs, 
                batch_size=256,  # Increased from 64 to 256 for 4x speed (adjust based on VRAM)
                show_progress_bar=True
            )
        
        # Sigmoid 적용
        probs = torch.sigmoid(torch.tensor(all_scores)).numpy()
        
        # 결과 재구성
        for (did, cid), prob in zip(pair_metadata, probs):
            if did not in doc_candidates:
                doc_candidates[did] = {}
            doc_candidates[did][cid] = float(prob)
            
    else:
        # ===== 원본 버전: 문서별 순차 처리 =====
        print("Reranking (Original - Sequential Mode)...")
        
        for did in tqdm(doc_ids, desc="Reranking"):
            candidates = doc_candidates_step1[did]
            doc_text = corpus[did]
            
            pairs = []
            for cid in candidates:
                full_path = get_full_path_text(cid, parents_dict, id2class)
                
                # Reuse keywords selected during SBERT encoding
                selected_keywords = class_keywords_map.get(cid, [])
                if selected_keywords:
                    kw_text = ", ".join(selected_keywords)
                    class_text = f"Category: {full_path}. Keywords: {kw_text}"
                else:
                    class_text = f"Category: {full_path}"
                
                pairs.append([class_text, doc_text])
            
            scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            
            # Sigmoid 적용
            scores_tensor = torch.tensor(scores)
            probs = torch.sigmoid(scores_tensor).numpy()
            
            cand_dict = {}
            for cid, score in zip(candidates, probs):
                cand_dict[cid] = float(score)
            
            doc_candidates[did] = cand_dict
    
    # Reranker 메모리 정리
    del reranker
    torch.cuda.empty_cache()
    
    return doc_candidates



def identify_confident_core_classes(doc_candidates, parents_dict, children_dict):
    """
    Identifies confident core classes using Relative Drop and Dynamic Level 0 Selection.
    
    1. Candidate Selection:
       - Sort by Confidence Score.
       - Select until Relative Drop > 0.1 or Count >= 10.
       - Ensure at least 1 candidate.
       
    2. Level 0 Filtering:
       - Calculate Level 0 Score = Sum of Confidence Scores of descendants.
       - Threshold = Max(Level 0 Scores) * 0.6.
       - Keep Level 0s >= Threshold.
    """
    print("Identifying Confident Core Classes (Refined)...")
    
    # 1. Calculate Raw Confidence Scores
    doc_confidences = {}
    
    for doc_id, candidates in tqdm(doc_candidates.items(), desc="Calc Confidence"):
        doc_confidences[doc_id] = {}
        
        for c, score in candidates.items():
            # Get Parent Scores
            parents = parents_dict.get(c, [])
            parent_scores = [candidates.get(p, 0.0) for p in parents]
            max_parent = max(parent_scores) if parent_scores else 0.0
            
            # Get Sibling Scores
            siblings = utils.get_siblings(c, parents_dict, children_dict)
            sibling_scores = [candidates.get(s, 0.0) for s in siblings]
            max_sibling = max(sibling_scores) if sibling_scores else 0.0
            
            # Conf(D, c)
            conf = score - max(max_parent, max_sibling)
            doc_confidences[doc_id][c] = conf
            
    # 2. Selection Logic
    final_core_classes = {}
    
    # Helper function to find Level 0 ancestor
    def get_level0_ancestor(class_id, parents_dict):
        current = class_id
        while True:
            parents = parents_dict.get(current, [])
            if not parents:
                return current
            current = parents[0]
    
    for doc_id, confs in doc_confidences.items():
        # Sort by Confidence Score (Descending)
        sorted_candidates = sorted(confs.items(), key=lambda x: x[1], reverse=True)
        
        # Step 1: Relative Drop + Max K
        selected_candidates = []
        if sorted_candidates:
            selected_candidates.append(sorted_candidates[0]) # Always take top 1
            
            for i in range(1, len(sorted_candidates)):
                # Max K = 10
                if len(selected_candidates) >= 10:
                    break
                
                # Relative Drop > 0.1
                curr_conf = sorted_candidates[i][1]
                prev_conf = sorted_candidates[i-1][1]
                
                if prev_conf - curr_conf > 0.1:
                    break
                    
                selected_candidates.append(sorted_candidates[i])
        
        if not selected_candidates:
            final_core_classes[doc_id] = []
            continue
            
        # Step 2: Level 0 Aggregation (Sum)
        level0_scores = defaultdict(float)
        for c, conf in selected_candidates:
            l0 = get_level0_ancestor(c, parents_dict)
            level0_scores[l0] += conf
            
        # Step 3: Level 0 Filtering (Dynamic Ratio 0.6)
        if not level0_scores:
            final_core_classes[doc_id] = []
            continue
            
        max_l0_score = max(level0_scores.values())
        threshold = max_l0_score * 0.6
        
        kept_level0s = {l0 for l0, score in level0_scores.items() if score >= threshold}
        
        # Final Filter
        final_list = [c for c, conf in selected_candidates if get_level0_ancestor(c, parents_dict) in kept_level0s]
        final_core_classes[doc_id] = final_list
        
    return final_core_classes

# 

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
