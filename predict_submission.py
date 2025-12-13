import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import src.utils as utils
import src.models as models
import src.trainer as trainer

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    TRAIN_DATA = 'Amazon_products/train/train_corpus.txt'
    TEST_DATA = 'Amazon_products/test/test_corpus.txt'
    TAXONOMY = 'Amazon_products/class_hierarchy.txt'
    CHECKPOINT_DIR = 'checkpoints'
    
    # 1. Load Data
    # 1. Load Data
    print("Loading data...")
    
    # Load Train Corpus
    train_corpus = {}
    with open(TRAIN_DATA, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                doc_id = parts[0]
                text = " ".join(parts[1:])
                train_corpus[doc_id] = text
                
    # Load Test Corpus
    test_corpus = {}
    with open(TEST_DATA, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                doc_id = parts[0]
                text = " ".join(parts[1:])
                test_corpus[doc_id] = text
    
    # Extract Classes & Build Graph
    print("Building Graph...")
    classes = set()
    edges = []
    with open(TAXONOMY, 'r') as f:
        for line in f:
            parent, child = line.strip().split('\t')
            classes.add(parent)
            classes.add(child)
            edges.append((parent, child))
            
    sorted_classes = sorted(list(classes))
    class2id = {c: i for i, c in enumerate(sorted_classes)}
    id2class = {i: c for i, c in enumerate(sorted_classes)}
    
    # 2. Initialize Model Components (Required for Constructor)
    print("Initializing Model Components...")
    
    # Recompute Initial Label Embeddings (Needed for Model Constructor)
    print("Computing initial label embeddings...")
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    
    label_emb_init = torch.zeros(len(id2class), 768).to(device)
    class_names = [id2class[i] for i in range(len(id2class))]
    batch_size_emb = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(class_names), batch_size_emb), desc="Label Embeddings"):
            batch_names = class_names[i:i+batch_size_emb]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = bert_model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            label_emb_init[i:i+batch_size_emb] = embs
            
    # Build Adjacency Matrix
    adj = utils.build_adjacency_matrix(len(id2class), edges).to(device)
    
    # 3. Initialize Model
    print("Initializing TaxoClassModel...")
    model = models.TaxoClassModel(len(id2class), label_emb_init, adj, model_name="microsoft/deberta-v3-base").to(device)
    
    # 4. Load Checkpoint
    # Find the latest checkpoint
    latest_ckpt = None
    for i in range(30, 0, -1): # Check up to 10 iterations
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'warmup_epoch_{i}.pth')
        if os.path.exists(ckpt_path):
            latest_ckpt = ckpt_path
            print(f"Found latest checkpoint: {latest_ckpt}")
            break
            
    if latest_ckpt:
        print(f"Loading checkpoint from {latest_ckpt}...")
        # Fix for sparse tensor loading issue: strict=False
        # The 'adj' buffer is sparse and might cause issues during loading if strict=True
        # Since 'adj' is re-initialized in the constructor, we can skip loading it if needed.
        # However, strict=False will load everything else matching.
        state_dict = torch.load(latest_ckpt, map_location=device)
        # Remove 'adj' from state_dict if it causes issues (it's a buffer, usually fixed)
        if 'adj' in state_dict:
            del state_dict['adj']
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint found! Please train the model first.")
        return

    # 5. Generate Predictions
    print("Generating predictions for Test Set...")
    test_doc_ids = sorted(list(test_corpus.keys()))
    
    # Create Dataset & DataLoader
    test_dataset = trainer.TextDataset(test_doc_ids, test_corpus, bert_tokenizer, max_len=128)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Predict
    logits, _ = trainer.predict(model, test_dataloader, device)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # 6. Format Submission
    print("Formatting submission...")
    predictions = []
    for i in range(len(test_doc_ids)):
        # Adaptive Thresholding Logic
        p = probs[i]
        selected = np.where(p > 0.1)[0]
        
        if len(selected) < 2:
            selected = np.argsort(p)[-2:]
        elif len(selected) > 3:
            top3_indices = np.argsort(p)[-3:]
            selected = top3_indices
            
        selected = sorted(selected)
        # Format: "10,64,93" (quoted automatically by pandas if needed, but sample shows quotes)
        pred_str = ",".join([str(idx) for idx in selected])
        predictions.append(pred_str)
        
    # Save to CSV
    # Format: id, label
    df = pd.DataFrame({'id': test_doc_ids, 'label': predictions})
    df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
