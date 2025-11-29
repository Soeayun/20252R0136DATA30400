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

def self_training_loop(model, train_corpus, test_corpus, tokenizer, 
                       initial_targets, initial_masks, 
                       parents_dict, children_dict, num_classes,
                       device, num_iterations=3, epochs_per_iter=3, batch_size=32, lr=2e-5):
    
    # Combine corpus for prediction (Transductive setting allows using Test data features)
    # But we only train on Train corpus (with Silver Labels).
    # Wait, TaxoClass usually trains on Unlabeled Data (Train) using Pseudo Labels.
    # We can also generate Pseudo Labels for Test data and train on it?
    # The prompt says "perform product review classification without using any labeled data".
    # And "You are allowed to use the test corpus information during training".
    # So we can treat Test Corpus as Unlabeled Data too and include it in Self-Training!
    
    # For simplicity, let's start with Train Corpus only for training, 
    # but we can expand to Test Corpus if needed.
    
    doc_ids = sorted(list(train_corpus.keys()))
    
    current_targets = initial_targets
    current_masks = initial_masks
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for iteration in range(num_iterations):
        print(f"\n=== Self-Training Iteration {iteration + 1}/{num_iterations} ===")
        
        # 1. Create Dataset & DataLoader
        dataset = TextDataset(doc_ids, train_corpus, tokenizer, current_targets, current_masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Train
        for epoch in range(epochs_per_iter):
            loss = train_epoch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch+1}/{epochs_per_iter} - Loss: {loss:.4f}")
            
        # 3. Re-predict
        # Create dataloader without shuffle for prediction
        pred_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logits, _ = predict(model, pred_dataloader, device)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # 4. Update Core Classes & Labels
        # Strategy: Select Top-1 (or Top-K with threshold) as new Core Class
        # TaxoClass uses "High Confidence" selection.
        # Simple approach: Top-1 for every doc.
        new_core_classes = np.argsort(probs, axis=1)[:, -1:]
        
        # Expand labels again
        # Note: We need to import expand_labels from core_mining. 
        # But to avoid circular import, we should pass the function or move it to utils.
        # I'll assume it's available or moved.
        # Let's import it inside the function or assume main passes the logic.
        from src.core_mining import expand_labels
        
        new_targets, new_masks = expand_labels(new_core_classes, parents_dict, children_dict, num_classes)
        
        current_targets = new_targets
        current_masks = new_masks
        
    return model
