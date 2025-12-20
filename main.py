import os
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from src import utils, core_mining, models, trainer

def main():
    # 1. Setup
    utils.set_seed(42)
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # Paths
    DATA_DIR = "Amazon_products"
    CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
    HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
    KEYWORDS_PATH = os.path.join(DATA_DIR, "class_related_keywords.txt")
    TRAIN_PATH = os.path.join(DATA_DIR, "train", "train_corpus.txt")
    TEST_PATH = os.path.join(DATA_DIR, "test", "test_corpus.txt")
    
    # 2. Load Data
    print("Loading data...")
    id2class, class2id = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    class2keywords = utils.load_keywords(KEYWORDS_PATH)
    #train_corpus = utils.load_corpus(TRAIN_PATH)
    train_corpus = utils.load_corpus(TEST_PATH)
    
    num_classes = len(id2class)
    print(f"Loaded {num_classes} classes.")
    #print(f"Loaded {len(train_corpus)} training docs.")
    print(f"Loaded {len(train_corpus)} test docs.")
    
    # 3. Build Graph
    print("Building Graph...")
    adj = utils.build_adjacency_matrix(num_classes, edges).to(device)
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, num_classes)
    
    # 4. Core Class Mining (Train Corpus)
    # Note: We can also use Test Corpus for Transductive Learning (Self-Training on Test)
    # For now, let's focus on Train Corpus for Silver Label Generation.
    
    # --- 4. Core Class Mining (Unified Pipeline) ---
    # Cache paths
    DOC_CANDIDATES_CACHE = os.path.join("checkpoints", "doc_candidates.json")
    CORE_CLASSES_CACHE = os.path.join("checkpoints", "core_classes.json")
    CORE_CLASSES_LLM_CACHE = os.path.join("checkpoints", "core_classes_llm_refined.json")
    AMB_IDS_CACHE = os.path.join("checkpoints", "ambiguous_doc_ids.json")
    
    train_doc_ids = sorted(list(train_corpus.keys()))
    
    # Create checkpoints directory if not exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- Step 4.1: Generate Doc Candidates (SBERT + Reranker) ---
    if os.path.exists(DOC_CANDIDATES_CACHE):
        print(f"[Step 4.1] Loading Doc Candidates from {DOC_CANDIDATES_CACHE}...")
        with open(DOC_CANDIDATES_CACHE, 'r') as f:
            loaded_candidates = json.load(f)
            doc_candidates = {int(k): {int(ck): cv for ck, cv in v.items()} for k, v in loaded_candidates.items()}
    else:
        print("[Step 4.1] Generating Doc Candidates (SBERT + Reranker)...")
        doc_candidates = core_mining.generate_core_classes_sbert_reranker(
            train_corpus, id2class, train_doc_ids, parents_dict, children_dict, device,
            sbert_model_name="BAAI/bge-m3",
            reranker_model_name="BAAI/bge-reranker-v2-m3",
            batch_size=32,
            class2keywords=class2keywords
        )
        with open(DOC_CANDIDATES_CACHE, 'w') as f:
            json.dump(doc_candidates, f)
        print(f"  ✓ Saved to {DOC_CANDIDATES_CACHE}")
    
    # --- Step 4.2: Identify Core Classes ---
    if os.path.exists(CORE_CLASSES_CACHE):
        print(f"[Step 4.2] Loading Core Classes from {CORE_CLASSES_CACHE}...")
        with open(CORE_CLASSES_CACHE, 'r') as f:
            confident_core_classes = {int(k): v for k, v in json.load(f).items()}
        # Load ambiguous doc IDs if exists
        if os.path.exists(AMB_IDS_CACHE):
            with open(AMB_IDS_CACHE, 'r') as f:
                ambiguous_doc_ids = json.load(f)
        else:
            ambiguous_doc_ids = []
    else:
        print("[Step 4.2] Identifying Core Classes...")
        confident_core_classes, ambiguous_doc_ids = core_mining.identify_confident_core_classes(
            doc_candidates, parents_dict, children_dict
        )
        with open(CORE_CLASSES_CACHE, 'w') as f:
            json.dump(confident_core_classes, f)
        with open(AMB_IDS_CACHE, 'w') as f:
            json.dump(ambiguous_doc_ids, f)
        print(f"  ✓ Saved {len(confident_core_classes)} docs to {CORE_CLASSES_CACHE}")
        print(f"  ✓ Saved {len(ambiguous_doc_ids)} ambiguous doc IDs to {AMB_IDS_CACHE}")
    
    # --- Step 4.3: LLM Refinement (REQUIRED) ---
    if os.path.exists(CORE_CLASSES_LLM_CACHE):
        print(f"[Step 4.3] Loading LLM-refined Core Classes from {CORE_CLASSES_LLM_CACHE}...")
        with open(CORE_CLASSES_LLM_CACHE, 'r') as f:
            core_classes_dict = {int(k): v for k, v in json.load(f).items()}
    else:
        # LLM refinement is REQUIRED - check for API key
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            print("\n" + "=" * 80)
            print("❌ ERROR: OPENAI_API_KEY not found!")
            print("=" * 80)
            print("LLM refinement is REQUIRED to generate core_classes_llm_refined.json")
            print("\nPlease create a .env file with your OpenAI API key:")
            print('  echo "OPENAI_API_KEY=your-key-here" > .env')
            print("\nThen run main.py again.")
            print("=" * 80)
            import sys
            sys.exit(1)
        
        print(f"[Step 4.3] Running LLM Refinement on {len(ambiguous_doc_ids)} ambiguous docs...")
        print("  ⚠ This is REQUIRED and will use API credits.")
        
        from src.llm_refinement import refine_core_classes_with_llm
        
        refined_core_classes, llm_decisions = refine_core_classes_with_llm(
            core_classes_dict=confident_core_classes,
            doc_candidates=doc_candidates,
            corpus=train_corpus,
            id2class=id2class,
            ambiguous_doc_ids=ambiguous_doc_ids,
            edges=edges,
            max_api_calls=1000,
            batch_size=20,
            max_parallel=5
        )
        
        core_classes_dict = refined_core_classes
        
        with open(CORE_CLASSES_LLM_CACHE, 'w') as f:
            json.dump(core_classes_dict, f)
        print(f"  ✓ Saved LLM-refined classes to {CORE_CLASSES_LLM_CACHE}")
    
    # --- Post-processing: Limit Level 0 to max 2 ---
    print("[Step 4.4] Post-processing: Limiting Level 0 classes to max 2...")
    filtered_count = 0
    for doc_id, classes in core_classes_dict.items():
        if not classes or len(classes) == 0:
            continue
        
        # Find Level 0 ancestor for each class
        level0_to_classes = {}
        for cid in classes:
            lv0 = core_mining.find_level0_ancestor(cid, parents_dict)
            if lv0 not in level0_to_classes:
                level0_to_classes[lv0] = []
            level0_to_classes[lv0].append(cid)
        
        # If more than 2 Level 0, keep top 2 by class count
        if len(level0_to_classes) > 2:
            sorted_lv0 = sorted(level0_to_classes.items(), key=lambda x: len(x[1]), reverse=True)[:2]
            top2_lv0_ids = {lv0 for lv0, _ in sorted_lv0}
            filtered_classes = [c for c in classes if core_mining.find_level0_ancestor(c, parents_dict) in top2_lv0_ids]
            core_classes_dict[doc_id] = filtered_classes
            filtered_count += 1
    
    print(f"  → Filtered {filtered_count} documents with 3+ Level 0 classes")
    
    # Convert to list of lists for next steps
    core_classes = []
    for doc_id in train_doc_ids:
        core_classes.append(core_classes_dict.get(doc_id, []))

  
    
    # --- 5.5. Label Expansion ---
    targets, masks = core_mining.expand_labels(core_classes, parents_dict, children_dict, num_classes)
    
    # --- 5.6 Filter documents with Core Classes ---
    # Only train on documents that have at least one core class (after Level 0 filtering)
    print("Filtering documents with Core Classes...")
    valid_indices = [i for i, cores in enumerate(core_classes) if len(cores) > 0]
    filtered_doc_ids = [train_doc_ids[i] for i in valid_indices]
    filtered_targets = targets[valid_indices]
    filtered_masks = masks[valid_indices]
    filtered_corpus = {doc_id: train_corpus[doc_id] for doc_id in filtered_doc_ids}
    
    print(f"Original training docs: {len(train_doc_ids)}")
    print(f"Docs with Core Classes: {len(filtered_doc_ids)} ({len(filtered_doc_ids)/len(train_doc_ids)*100:.1f}%)")
    print(f"Docs without Core Classes (excluded): {len(train_doc_ids) - len(filtered_doc_ids)}")
    
    # --- 6. Initialize Model ---
    print("Initializing Model...")
    # Initial Label Embeddings: Use BERT embeddings of class names
    # We compute this on the fly
    print("Computing initial label embeddings...")
    # Use the same model for label embedding initialization
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    
    label_emb_init = torch.zeros(len(id2class), 768).to(device)
    
    # Batch process label embeddings
    class_names = [id2class[i] for i in range(len(id2class))]
    batch_size_emb = 64
    
    with torch.no_grad():
        for i in range(0, len(class_names), batch_size_emb):
            batch_names = class_names[i:i+batch_size_emb]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = bert_model(**inputs)
            # Use [CLS] embedding (first token)
            embs = outputs.last_hidden_state[:, 0, :]
            label_emb_init[i:i+batch_size_emb] = embs
            
    # Build Adjacency Matrix
    adj = utils.build_adjacency_matrix(len(id2class), edges).to(device)
    
    # Create Full Model
    model = models.TaxoClassModel(len(id2class), label_emb_init, adj, model_name="microsoft/deberta-v3-base", use_gat=True).to(device)
    
    # --- 6.5 Supervised Warm-up (Step 3) ---
    # Train on Silver Labels (Core Classes) first to avoid Mode Collapse
    print("Starting Supervised Warm-up...") 
    model = trainer.supervised_training_loop(
        model, filtered_corpus, bert_tokenizer, 
        filtered_targets, filtered_masks, device, 
        epochs=40, batch_size=64, lr=5e-5
    )

    # --- 7. Iterative Self-Training with Unlabeled Documents ---
    print("\n" + "="*80)
    print("Iterative Self-Training Phase")
    print("="*80)
    
    from src import self_training
    
    # Configuration
    NUM_ITERATIONS = 1  # Number of self-training iterations
    EPOCHS_PER_ITERATION = 3  # Epochs for each re-training
    iteration = 0  # Initialize for print statement at the end
    
    # Start with warmup data as labeled dataitera
    current_labeled_doc_ids = filtered_doc_ids
    current_labeled_corpus = filtered_corpus
    current_labeled_targets = filtered_targets
    current_labeled_masks = filtered_masks
    
    # Track already labeled documents (warmup + pseudo from all iterations)
    already_labeled_set = set(filtered_doc_ids)
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*80}")
        print(f"Self-Training Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*80}")
        
        # Identify remaining unlabeled documents
        unlabeled_doc_ids = [did for did in train_doc_ids if did not in already_labeled_set]
        
        if len(unlabeled_doc_ids) == 0:
            print("\n⚠️  No more unlabeled documents. Stopping self-training.")
            break
        
        print(f"\n📊 Remaining Unlabeled: {len(unlabeled_doc_ids):,} ({len(unlabeled_doc_ids)/len(train_doc_ids)*100:.1f}%)")
        print(f"   Already Labeled: {len(already_labeled_set):,}")
        
        unlabeled_corpus = {did: train_corpus[did] for did in unlabeled_doc_ids}
        
        # Perform self-training iteration
        combined_doc_ids, combined_corpus, combined_targets, combined_masks, stats, pseudo_predictions = \
            self_training.self_training_iteration(
                model=model,
                tokenizer=bert_tokenizer,
                device=device,
                train_doc_ids=current_labeled_doc_ids,
                train_corpus=current_labeled_corpus,
                train_targets=current_labeled_targets,
                train_masks=current_labeled_masks,
                unlabeled_doc_ids=unlabeled_doc_ids,
                unlabeled_corpus=unlabeled_corpus,
                parents_dict=parents_dict,
                children_dict=children_dict,
                num_classes=len(id2class),
                min_threshold=0.85,
                min_num_classes=2,
                batch_size=32,
                verbose=True
            )
        
        # Save pseudo-label predictions for this iteration
        PSEUDO_LABELS_PATH = f"checkpoints/pseudo_labels_iter{iteration+1}.json"
        with open(PSEUDO_LABELS_PATH, 'w') as f:
            json.dump(pseudo_predictions, f, indent=2)
        print(f"\n💾 Saved iteration {iteration+1} pseudo-labels to {PSEUDO_LABELS_PATH}")
        
        # Re-train if pseudo-labels were generated
        if stats.get('num_pseudo', 0) > 0:
            print(f"\n{'='*80}")
            print(f"Re-training with Combined Dataset (Iteration {iteration+1})")
            print(f"{'='*80}")
            
            model = trainer.supervised_training_loop(
                model, combined_corpus, bert_tokenizer,
                combined_targets, combined_masks, device,
                epochs=EPOCHS_PER_ITERATION,
                batch_size=64,
                lr=2e-5,
                checkpoint_prefix=f'retrain_iter{iteration+1}'
            )
            
            # Update labeled data for next iteration
            current_labeled_doc_ids = combined_doc_ids
            current_labeled_corpus = combined_corpus
            current_labeled_targets = combined_targets
            current_labeled_masks = combined_masks
            
            # Update already labeled set
            already_labeled_set.update(combined_doc_ids)
            
            print(f"\n✅ Iteration {iteration+1} completed!")
            print(f"   Pseudo-labels added: +{stats['num_pseudo']:,}")
            print(f"   Total labeled: {len(already_labeled_set):,} ({len(already_labeled_set)/len(train_doc_ids)*100:.1f}%)")
            print(f"   Selection rate: {stats['selection_rate']*100:.1f}%")
        else:
            print(f"\n⚠️  No pseudo-labels met the criteria in iteration {iteration+1}. Stopping.")
            break
    
    print(f"\n{'='*80}")
    print(f"Train Corpus Self-Training Complete - {iteration+1} iterations")
    print(f"{'='*80}")

    # --- 8. Test Corpus Self-Training ---
    # (Commented out - not used in current pipeline)
    
    #model, current_labeled_doc_ids, current_labeled_corpus, current_labeled_targets, current_labeled_masks, used_test_doc_ids = \
    #test_corpus_training.iterative_test_corpus_training(
    #    model=model,
    #    tokenizer=bert_tokenizer,
    #    device=device,
    #    current_labeled_doc_ids=current_labeled_doc_ids,
    #    current_labeled_corpus=current_labeled_corpus,
    #    current_labeled_targets=current_labeled_targets,
    #    current_labeled_masks=current_labeled_masks,
    #    test_corpus=test_corpus,
    #    parents_dict=parents_dict,
    #    children_dict=children_dict,
    #    num_classes=len(id2class),
    #    num_iterations=3,      # Number of test corpus iterations
    #    docs_per_iteration=300,  # Documents per iteration
    #    threshold=0.85,         # Lower threshold for broader pool, random sampling
    #    epochs=3,              # Training epochs per iteration
    #    batch_size=64,
    #    lr=2e-5
    #)
    
    # METHOD 2: TaxoClass Original (Soft Selection with KL Divergence)
    # (Commented out - not used in current pipeline)
    
   # model = test_taxoclass_training.taxoclass_test_corpus_training(
   #     model=model,
   #     tokenizer=bert_tokenizer,
   #     device=device,
   #     current_labeled_corpus=current_labeled_corpus,
   #     test_corpus=test_corpus,
   #     num_iterations=1,      # More iterations for frequent Q updates
   #     epochs_per_iter=1,     # Fewer epochs to prevent stale Q
   #     batch_size=32,
   #     lr=1e-5
   # )
    
    print(f"\n{'='*80}")
    print(f"All Self-Training Complete")
    print(f"{'='*80}")
    
    # Save Final Model
    MODEL_PATH = "checkpoints/taxoclass_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")
    
    # 7. Final Prediction on Test Set
    print("Generating predictions for Test Set...")
    test_doc_ids = sorted(list(train_corpus.keys()))
    test_dataset = trainer.TextDataset(test_doc_ids, train_corpus, bert_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logits, _ = trainer.predict(model, test_dataloader, device)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Format submission
    # We need to output predicted class IDs.
    # Thresholding? Or Top-K?
    # Project description says: "each document is associated with at least two and at most three labels."
    # We can pick Top-3? Or use a threshold.
    # Let's use Top-3 for now as a baseline.
    
    predictions = []
    for i in range(len(test_doc_ids)):
        # Adaptive Thresholding
        # 1. Select classes with prob > 0.5
        p = probs[i]
        selected = np.where(p > 0.65)[0]
        
        # 2. Constraints: At least 2, At most 3
        if len(selected) < 2:
            # Fallback: Top-2 (Requirement: at least 2 labels)
            selected = np.argsort(p)[-2:]
        elif len(selected) > 3:
            # Limit to Top-3
            top3_indices = np.argsort(p)[-3:]
            # Intersect selected with top3 to keep order or just take top3
            selected = top3_indices
            
        # Convert to class IDs (which are just indices here)
        # Sort indices to be deterministic
        selected = sorted(selected)
        pred_str = ",".join([str(idx) for idx in selected])
        predictions.append(pred_str)
        
    # Save to CSV
    # Format: pid, labels
    df = pd.DataFrame({'id': test_doc_ids, 'label': predictions})
    df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
