import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class TextDataset(Dataset):
    def __init__(self, doc_ids, corpus, tokenizer, targets=None, masks=None, max_len=256):
        self.doc_ids = doc_ids
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.targets = targets
        self.masks = masks
        self.max_len = max_len
        
    def __len__(self):
        return len(self.doc_ids)
    
    def __getitem__(self, idx):
        did = self.doc_ids[idx]
        text = self.corpus[did]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'doc_id': did
        }
        
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
            if self.masks is not None:
                item['masks'] = torch.tensor(self.masks[idx], dtype=torch.float)
            
        return item

class HierarchicalBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, logits, targets, masks):
        loss = self.bce(logits, targets)
        # Apply mask
        masked_loss = loss * masks
        # Average over valid elements
        return masked_loss.sum() / masks.sum()

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    criterion = HierarchicalBCELoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, targets, masks)
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def predict(model, dataloader, device):
    model.eval()
    all_logits = []
    all_doc_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            # Handle doc_id: it might be a list (from default collate) or tensor
            doc_ids = batch['doc_id']
            if isinstance(doc_ids, torch.Tensor):
                all_doc_ids.extend(doc_ids.tolist())
            else:
                all_doc_ids.extend(doc_ids)
            
    return np.vstack(all_logits), all_doc_ids

def evaluate(model, dataloader, device, true_labels=None):
    """
    Evaluation function.
    Note: Since we don't have true labels for Train set (unlabeled), 
    this is mostly for Test set if labels were available, or monitoring.
    For this project, we might not have ground truth for evaluation during training 
    unless we use a validation split from Silver Labels (which are noisy).
    """
    # Placeholder for now
    pass

def supervised_training_loop(model, train_corpus, tokenizer, targets, masks, device, epochs=3, batch_size=32, lr=5e-5):
    """
    Step 3 of TaxoClass: Train classifier f(.) with Eq. (8) (BCE Loss)
    This serves as a warm-up phase using Silver Labels (Core Classes).
    Includes checkpointing to resume training.
    """
    print("\n=== Starting Supervised Warm-up (Silver Labels) ===")
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    # Check for existing checkpoints
    for i in range(epochs, 0, -1):
        ckpt_path = os.path.join(checkpoint_dir, f'warmup_epoch_{i}.pth')
        if os.path.exists(ckpt_path):
            print(f"Found warm-up checkpoint: {ckpt_path}. Resuming from Epoch {i+1}...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = i
            break
            
    if start_epoch >= epochs:
        print("Warm-up already completed.")
        return model
    
    doc_ids = sorted(list(train_corpus.keys()))
    dataset = TextDataset(doc_ids, train_corpus, tokenizer, targets=targets, masks=masks, max_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(start_epoch, epochs):
        # Use train_epoch which uses HierarchicalBCELoss (handling masks)
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f'warmup_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved warm-up checkpoint: {ckpt_path}")
        
    print("=== Supervised Warm-up Completed ===\n")
    return model

def self_training_loop(model, train_corpus, test_corpus, tokenizer, 
                       initial_targets, initial_masks, 
                       parents_dict, children_dict, num_classes,
                       device, num_iterations=3, epochs_per_iter=3, batch_size=32, lr=1e-5):
    
    print("\n=== Starting Multi-label Self-Training ===")
    
    # Combine Train and Test corpus for Self-Training (Transductive)
    # TaxoClass uses Unlabeled Data (D) which includes everything available.
    train_doc_ids = sorted(list(train_corpus.keys()))
    test_doc_ids = sorted(list(test_corpus.keys()))
    all_doc_ids = train_doc_ids + test_doc_ids
    
    # Create a unified corpus dict
    all_corpus = {**train_corpus, **test_corpus}
    
    # Initial Training Data (only from Train set with Core Classes)
    # We start by training on the initial silver labels (Core Classes)
    # This is already done in main.py before calling this loop?
    # Actually, main.py calls this AFTER initial training.
    # So 'model' is already trained on Core Classes.
    
    # We need to generate predictions P for ALL data
    full_dataset = TextDataset(all_doc_ids, all_corpus, tokenizer, max_len=128)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    # Checkpoint Resume Logic
    start_iteration = 0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoints
    for i in range(num_iterations - 1, -1, -1):
        ckpt_path = os.path.join(checkpoint_dir, f'st_iter_{i+1}.pth')
        if os.path.exists(ckpt_path):
            print(f"Found checkpoint: {ckpt_path}. Resuming from Iteration {i+2}...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_iteration = i + 1
            break
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Use BCEWithLogitsLoss for stability and to ensure non-negative loss
    # This is mathematically equivalent to minimizing KL(Q || P) since Q is constant wrt P
    st_criterion = nn.BCEWithLogitsLoss()
    
    for iteration in range(start_iteration, num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # 1. Predict P on all data
        logits, _ = predict(model, full_dataloader, device)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # 2. Calculate Class Frequencies (Cached Statistics)
        # Instead of fixing Q for the whole epoch, we fix the normalization factor f_j = sum_i p_ij
        # and update Q dynamically for each batch using the current model predictions.
        print("Calculating Class Frequencies (Cached Statistics)...")
        # probs shape: (N, C)
        # f_j shape: (C,)
        f_j = probs.sum(axis=0)
        
        # Avoid division by zero
        f_j = np.maximum(f_j, 1e-8)
        
        # Convert to torch tensor for use in training loop
        f_j_tensor = torch.tensor(f_j, dtype=torch.float).to(device)
        
        # 3. Train with Dynamic Q
        # We iterate over the dataset, but we don't use the pre-calculated targets_q.
        # Instead, we calculate Q on the fly.
        
        # Create dataset without targets (we compute them on the fly)
        st_dataset = TextDataset(all_doc_ids, all_corpus, tokenizer, targets=None, masks=None, max_len=128)
        st_dataloader = DataLoader(st_dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        total_loss = 0
        
        for epoch in range(epochs_per_iter):
            epoch_loss = 0
            for batch in tqdm(st_dataloader, desc=f"ST Training Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                
                # --- Dynamic Q Calculation (Binary DEC with Cached Statistics) ---
                # P: Current predictions
                current_probs = torch.sigmoid(logits) # (Batch, C)
                
                # Q Calculation:
                # We adapt DEC formula for Multi-label (Binary) classification.
                # q_ij = (p_ij^2 / f_j) / ( (p_ij^2 / f_j) + ((1-p_ij)^2 / (N - f_j)) )
                # This balances sharpening based on class frequency f_j.
                
                N_docs = len(all_doc_ids)
                f_j_neg = N_docs - f_j_tensor
                f_j_neg = torch.maximum(f_j_neg, torch.tensor(1e-8).to(device))
                
                # Positive term: p^2 / f
                q_pos = (current_probs ** 2) / f_j_tensor
                
                # Negative term: (1-p)^2 / (N-f)
                q_neg = ((1 - current_probs) ** 2) / f_j_neg
                
                # Normalized Target Q
                batch_targets = q_pos / (q_pos + q_neg)
                
                # Calculate Loss
                loss = st_criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(st_dataloader)
            print(f"Iter {iteration+1}, Epoch {epoch+1} - ST Loss: {avg_loss:.4f}")
            
        # Save checkpoint after each iteration
        ckpt_path = os.path.join(checkpoint_dir, f'st_iter_{iteration+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
            
    return model
