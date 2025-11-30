import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
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
    train_corpus = utils.load_corpus(TRAIN_PATH)
    test_corpus = utils.load_corpus(TEST_PATH)
    
    num_classes = len(id2class)
    print(f"Loaded {num_classes} classes.")
    print(f"Loaded {len(train_corpus)} training docs.")
    print(f"Loaded {len(test_corpus)} test docs.")
    
    # 3. Build Graph
    print("Building Graph...")
    adj = utils.build_adjacency_matrix(num_classes, edges).to(device)
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, num_classes)
    
    # 4. Core Class Mining (Train Corpus)
    # Note: We can also use Test Corpus for Transductive Learning (Self-Training on Test)
    # For now, let's focus on Train Corpus for Silver Label Generation.
    
    # 4.1 BM25
    BM25_CACHE = "checkpoints/bm25_scores.npy"
    # --- 4. Core Class Mining (Hybrid Top-down) ---
    CORE_CLASSES_CACHE = os.path.join("checkpoints", "core_classes.json")
    train_doc_ids = sorted(list(train_corpus.keys())) # Ensure train_doc_ids is defined for the new section
    
    if os.path.exists(CORE_CLASSES_CACHE):
        print(f"Loading Core Classes from {CORE_CLASSES_CACHE}...")
        with open(CORE_CLASSES_CACHE, 'r') as f:
            # Load and convert keys back to int (JSON keys are strings)
            loaded_core = json.load(f)
            core_classes_dict = {int(k): v for k, v in loaded_core.items()} # Renamed to avoid conflict
            
            # We need a list of lists for compatibility with existing code
            # core_classes dict: {doc_id: [class_id1, ...]}
            # We need to ensure the order matches doc_ids
            core_classes = [] # This will be the list of lists
            for doc_id in train_doc_ids: # Use train_doc_ids
                core_classes.append(core_classes_dict.get(doc_id, []))
            
    else:
        print("Starting Core Class Mining (Hybrid Top-down)...")
        
        # 4.1 Candidate Selection (Hybrid Top-down)
        doc_candidates = core_mining.generate_core_classes_hybrid_top_down(
            train_corpus, id2class, train_doc_ids, parents_dict, children_dict, device, # Use train_corpus and train_doc_ids
            model_name="cross-encoder/nli-deberta-v3-base",
            batch_size=32,
            class2keywords=class2keywords
        )
        
        # 4.2 Confident Core Class Identification
        confident_core_classes = core_mining.identify_confident_core_classes(
            doc_candidates, parents_dict, children_dict
        )
        
        # Save checkpoints
        print(f"Saved Core Classes to {CORE_CLASSES_CACHE}")
        with open(CORE_CLASSES_CACHE, 'w') as f:
            json.dump(confident_core_classes, f)
            
        # Convert to list of lists for next steps
        core_classes = []
        for doc_id in train_doc_ids: # Use train_doc_ids
            core_classes.append(confident_core_classes.get(doc_id, []))

    # --- 5. Label Expansion --- # Renumbered from 4.4 to 5
    targets, masks = core_mining.expand_labels(core_classes, parents_dict, children_dict, num_classes) # Corrected num_classes
    
    # 6. Initialize Model # Renumbered from 5 to 6
    print("Initializing Model...")
    # Initial Label Embeddings: Use BERT embeddings of class names
    # We compute this on the fly
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    doc_encoder = models.DocumentEncoder().to(device) # Just to get the BERT model for embedding classes
    
    # Compute Class Embeddings
    print("Computing initial class embeddings...")
    class_texts = [id2class[i] for i in range(num_classes)]
    # Optional: Append keywords
    # class_texts = [f"{id2class[i]} {' '.join(class2keywords.get(id2class[i], []))}" for i in range(num_classes)]
    
    class_inputs = bert_tokenizer(class_texts, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    with torch.no_grad():
        # Use the bert model inside doc_encoder
        outputs = doc_encoder.bert(**class_inputs)
        label_emb_init = outputs.last_hidden_state[:, 0, :]
    
    # Create Full Model
    model = models.TaxoClassModel(num_classes, label_emb_init, adj).to(device)
    
    # 6. Self-Training
    print("Starting Self-Training...")
    model = trainer.self_training_loop(
        model, train_corpus, test_corpus, bert_tokenizer,
        targets, masks,
        parents_dict, children_dict, num_classes,
        device, num_iterations=3, epochs_per_iter=3
    )
    
    # Save Model
    MODEL_PATH = "checkpoints/taxoclass_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    
    # 7. Final Prediction on Test Set
    print("Generating predictions for Test Set...")
    test_doc_ids = sorted(list(test_corpus.keys()))
    test_dataset = trainer.TextDataset(test_doc_ids, test_corpus, bert_tokenizer)
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
        selected = np.where(p > 0.5)[0]
        
        # 2. Constraints: At least 1, At most 3
        if len(selected) == 0:
            # Fallback: Top-1
            selected = np.argsort(p)[-1:]
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
    df = pd.DataFrame({'pid': test_doc_ids, 'labels': predictions})
    df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
