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
    bm25_scores, train_doc_ids = core_mining.calculate_bm25_scores(train_corpus, class2keywords, id2class)
    
    # Filter Top-K for NLI (e.g., Top 10 to save time)
    top_k_bm25 = 10
    top_k_indices = np.argsort(bm25_scores, axis=1)[:, -top_k_bm25:]
    
    # 4.2 NLI
    # Warning: This takes time. 
    nli_scores = core_mining.calculate_entailment_scores(
        train_corpus, id2class, train_doc_ids, device, 
        top_k_filter=top_k_indices
    )
    
    # 4.3 Generate Silver Labels
    core_classes = core_mining.generate_silver_labels(bm25_scores, nli_scores, alpha=0.5, top_k=1)
    
    # 4.4 Expand Labels
    targets, masks = core_mining.expand_labels(core_classes, parents_dict, children_dict, num_classes)
    
    # 5. Initialize Model
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
        # Get indices of top 3
        top3 = np.argsort(probs[i])[-3:]
        # Convert to class IDs (which are just indices here)
        pred_str = " ".join([str(idx) for idx in top3])
        predictions.append(pred_str)
        
    # Save to CSV
    # Format: ID, Predicted
    df = pd.DataFrame({'ID': test_doc_ids, 'Predicted': predictions})
    df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
