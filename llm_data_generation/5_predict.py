"""
Step 5: Predict on test set using trained TaxoClass model

Uses the same prediction pipeline as main.py

Output: predictions.csv for Kaggle submission
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.append('..')
from src import utils, models, trainer

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


def main():
    print("="*70)
    print("PREDICTION ON TEST SET")
    print("="*70)
    
    # Paths
    DATA_DIR = "../Amazon_products"
    CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
    HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
    TEST_PATH = os.path.join(DATA_DIR, "test", "test_corpus.txt")
    MODEL_PATH = "taxoclass_synthetic.pth"
    
    # Load data
    print("\nLoading data...")
    id2class, class2id = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    test_corpus = utils.load_corpus(TEST_PATH)
    num_classes = len(id2class)
    
    print(f"Classes: {num_classes}")
    print(f"Test docs: {len(test_corpus)}")
    
    # Build adjacency matrix
    adj = utils.build_adjacency_matrix(num_classes, edges).to(DEVICE)
    
    # Initialize label embeddings (same as training)
    print("\nInitializing model...")
    from transformers import AutoModel
    
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(DEVICE)
    
    label_emb_init = torch.zeros(num_classes, 768).to(DEVICE)
    
    class_names = [id2class[i] for i in range(num_classes)]
    batch_size_emb = 64
    
    with torch.no_grad():
        for i in range(0, len(class_names), batch_size_emb):
            batch_names = class_names[i:i+batch_size_emb]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            outputs = bert_model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            label_emb_init[i:i+batch_size_emb] = embs
    
    # Load model
    model = models.TaxoClassModel(
        num_classes, label_emb_init, adj,
        model_name="microsoft/deberta-v3-base"
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✓ Model loaded")
    
    # Predict (same as main.py)
    print("\nGenerating predictions...")
    test_doc_ids = sorted(list(test_corpus.keys()))
    test_dataset = trainer.TextDataset(test_doc_ids, test_corpus, bert_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logits, _ = trainer.predict(model, test_dataloader, DEVICE)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Format predictions (same logic as main.py)
    predictions = []
    for i in range(len(test_doc_ids)):
        p = probs[i]
        selected = np.where(p > 0.5)[0]
        
        # Constraints: At least 2, At most 3
        if len(selected) < 2:
            selected = np.argsort(p)[-2:]
        elif len(selected) > 3:
            top3_indices = np.argsort(p)[-3:]
            selected = top3_indices
        
        selected = sorted(selected)
        pred_str = " ".join([str(idx) for idx in selected])
        predictions.append(pred_str)
    
    # Save to CSV (Kaggle format)
    df = pd.DataFrame({
        'id': test_doc_ids,
        'predict': predictions
    })
    df.to_csv('predictions.csv', index=False)
    
    print(f"\n✓ Saved {len(predictions)} predictions to predictions.csv")
    
    # Statistics
    pred_counts = [len(p.split()) for p in predictions]
    print(f"\nPrediction statistics:")
    print(f"  Avg classes per doc: {sum(pred_counts)/len(pred_counts):.2f}")
    print(f"  Min: {min(pred_counts)} | Max: {max(pred_counts)}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
