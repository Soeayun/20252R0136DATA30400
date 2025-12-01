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
            all_doc_ids.extend(batch['doc_id'].tolist())
            
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

class SelfTrainingLoss(nn.Module):
    """
    KL Divergence Loss for Self-Training.
    L = KL(Q || P) = sum(q * log(q/p))
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes) - Model predictions (before sigmoid)
        targets: (batch_size, num_classes) - Target distribution Q
        """
        probs = torch.sigmoid(logits)
        # Avoid log(0)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        targets = torch.clamp(targets, min=1e-7, max=1.0 - 1e-7)
        
        # KL Divergence: q * (log(q) - log(p))
        loss = targets * (torch.log(targets) - torch.log(probs))
        
        # Sum over classes, mean over batch
        return loss.sum(dim=1).mean()

def calculate_target_distribution(probs):
    """
    Calculates target distribution Q from current predictions P.
    t_ij = p_ij^2 / sum_i(p_ij) / (p_ij^2 / sum_i(p_ij) + (1-p_ij)^2 / sum_i(1-p_ij))
    """
    # probs: (num_docs, num_classes)
    
    # Avoid division by zero
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    
    # Calculate normalization terms (sum over documents)
    sum_p = np.sum(probs, axis=0, keepdims=True) # (1, num_classes)
    sum_1_p = np.sum(1 - probs, axis=0, keepdims=True)
    
    # Numerator: p^2 / sum_p
    numerator = probs**2 / sum_p
    
    # Denominator term 2: (1-p)^2 / sum_(1-p)
    denom_term2 = (1 - probs)**2 / sum_1_p
    
    # Q
    q = numerator / (numerator + denom_term2)
    
    return q

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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    st_criterion = SelfTrainingLoss()
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # 1. Predict P on all data
        logits, _ = predict(model, full_dataloader, device)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # 2. Calculate Target Distribution Q
        print("Updating Target Distribution Q...")
        targets_q = calculate_target_distribution(probs)
        
        # 3. Train on Q
        # We use Q as soft targets
        # Note: TaxoClass updates Q every 25 batches. 
        # Here we simplify by updating Q once per iteration (or epoch).
        # To strictly follow "every 25 batches", we would need to update Q dynamically inside the loop.
        # But calculating Q requires global statistics (sum over all docs).
        # So "every 25 batches" implies we either:
        # a) Update Q globally every 25 batches (expensive)
        # b) Or maybe they meant something else?
        # Re-reading: "In practice, instead of updating the target distribution for every training example, we update it every 25 batches"
        # This usually means Q is fixed for a few steps.
        # Given our dataset size (50k docs), updating Q every epoch is a reasonable approximation.
        
        # Create dataset with Soft Targets Q
        # We don't use masks here, as Q covers all classes
        st_dataset = TextDataset(all_doc_ids, all_corpus, tokenizer, targets=targets_q, masks=None, max_len=128)
        st_dataloader = DataLoader(st_dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        total_loss = 0
        
        for epoch in range(epochs_per_iter):
            epoch_loss = 0
            for batch in tqdm(st_dataloader, desc=f"ST Training Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_targets = batch['targets'].to(device) # Q
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                
                loss = st_criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(st_dataloader)
            print(f"Iter {iteration+1}, Epoch {epoch+1} - ST Loss: {avg_loss:.4f}")
            
    return model
