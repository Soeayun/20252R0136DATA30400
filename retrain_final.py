"""
Final Training Script

1. Load the trained model from retrain_iter2_epoch_3.pth
2. Generate pseudo-labels for test corpus with strict threshold (0.95, 2-3 classes)
3. Combine with core_classes_llm_refined.json data
4. Initialize a fresh model and train from scratch (supervised)
5. Save checkpoints to checkpoint_final/
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src import utils, core_mining, models, trainer

# Ensure output directory exists
os.makedirs("checkpoints_final", exist_ok=True)

def main():
    utils.set_seed(42)
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # ====== 1. Load Data ======
    print("\n" + "="*80)
    print("Step 1: Loading Data")
    print("="*80)
    
    # Paths (same as main.py)
    DATA_DIR = "Amazon_products"
    CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
    HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
    TRAIN_PATH = os.path.join(DATA_DIR, "train", "train_corpus.txt")
    TEST_PATH = os.path.join(DATA_DIR, "test", "test_corpus.txt")
    
    # Load taxonomy
    id2class, class2id = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    train_corpus = utils.load_corpus(TRAIN_PATH)
    test_corpus = utils.load_corpus(TEST_PATH)
    
    num_classes = len(id2class)
    train_doc_ids = sorted(list(train_corpus.keys()))
    test_doc_ids = sorted(list(test_corpus.keys()))
    
    # Build hierarchy relations
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, num_classes)
    
    print(f"Loaded {num_classes} classes.")
    print(f"Loaded {len(train_doc_ids)} train docs.")
    print(f"Loaded {len(test_doc_ids)} test docs.")
    
    # ====== 2. Load Trained Model ======
    print("\n" + "="*80)
    print("Step 2: Loading Trained Model")
    print("="*80)
    
    # Initialize components for model creation
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    bert_model_temp = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    
    # Compute label embeddings
    print("Computing initial label embeddings...")
    label_emb_init = torch.zeros(len(id2class), 768).to(device)
    class_names = [id2class[i] for i in range(len(id2class))]
    
    with torch.no_grad():
        for i in range(0, len(class_names), 64):
            batch_names = class_names[i:i+64]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = bert_model_temp(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            label_emb_init[i:i+64] = embs
    
    del bert_model_temp
    torch.cuda.empty_cache()
    
    # Build adjacency matrix
    adj = utils.build_adjacency_matrix(len(id2class), edges).to(device)
    
    # Create model and load checkpoint
    model = models.TaxoClassModel(len(id2class), label_emb_init, adj, 
                                   model_name="microsoft/deberta-v3-base").to(device)
    
    CHECKPOINT_PATH = "checkpoints/retrain_iter2_epoch_3.pth"
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Handle both formats: direct state_dict or wrapped in dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Loaded trained model.")
    
    # ====== 3. Generate Pseudo-Labels for Test Corpus ======
    print("\n" + "="*80)
    print("Step 3: Generating Pseudo-Labels for Test Corpus")
    print("="*80)
    
    # Create test dataloader
    test_dataset = trainer.TextDataset(test_doc_ids, test_corpus, bert_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Generate predictions
    print("Generating predictions for test corpus...")
    test_pseudo_predictions = {}
    
    threshold = 0.99
    min_classes = 2
    max_classes = 3
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Pseudo-labeling Test"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            doc_ids_batch = batch['doc_id']
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            for i, doc_id in enumerate(doc_ids_batch):
                doc_id_str = str(doc_id.item() if hasattr(doc_id, 'item') else doc_id)
                prob = probs[i].cpu().numpy()
                
                # Select classes above threshold
                selected_indices = np.where(prob >= threshold)[0]
                selected_probs = prob[selected_indices]
                
                # Sort by confidence
                sorted_idx = np.argsort(selected_probs)[::-1]
                selected_indices = selected_indices[sorted_idx]
                selected_probs = selected_probs[sorted_idx]
                
                # Limit to max_classes
                if len(selected_indices) > max_classes:
                    selected_indices = selected_indices[:max_classes]
                    selected_probs = selected_probs[:max_classes]
                
                # Check if accepted (2-3 classes above threshold)
                accepted = len(selected_indices) >= min_classes
                
                test_pseudo_predictions[doc_id_str] = {
                    'selected_classes': selected_indices.tolist(),
                    'confidences': selected_probs.tolist(),
                    'num_classes': len(selected_indices),
                    'accepted': accepted
                }
    
    # Count accepted
    accepted_count = sum(1 for pred in test_pseudo_predictions.values() if pred['accepted'])
    print(f"\n📊 Test Pseudo-Labeling Statistics:")
    print(f"   Total test docs: {len(test_doc_ids):,}")
    print(f"   Accepted (2+ classes, threshold {threshold}): {accepted_count:,} ({accepted_count/len(test_doc_ids)*100:.1f}%)")
    
    # Save pseudo-labels
    with open("checkpoints_final/pseudo_labels_test_final.json", 'w') as f:
        json.dump(test_pseudo_predictions, f, indent=2)
    print("Saved pseudo-labels to checkpoints_final/pseudo_labels_test_final.json")
    
    # ====== 4. Load Core Classes Data ======
    print("\n" + "="*80)
    print("Step 4: Loading Core Classes from core_classes_llm_refined.json")
    print("="*80)
    
    CORE_CLASSES_PATH = "checkpoints/core_classes_llm_refined.json"
    with open(CORE_CLASSES_PATH, 'r') as f:
        core_classes_dict = json.load(f)
    
    # Convert to list format
    core_classes = []
    for doc_id in train_doc_ids:
        doc_id_str = str(doc_id)
        core_classes.append(core_classes_dict.get(doc_id_str, []))
    
    print(f"Loaded core classes for {len(core_classes_dict):,} documents")
    
    # ====== 5. Expand Labels and Prepare Training Data ======
    print("\n" + "="*80)
    print("Step 5: Preparing Combined Training Data")
    print("="*80)
    
    # Expand train labels
    train_targets, train_masks = core_mining.expand_labels(
        core_classes, parents_dict, children_dict, num_classes
    )
    
    # Filter documents with core classes
    valid_indices = [i for i, cores in enumerate(core_classes) if len(cores) > 0]
    filtered_train_doc_ids = [train_doc_ids[i] for i in valid_indices]
    filtered_train_targets = train_targets[valid_indices]
    filtered_train_masks = train_masks[valid_indices]
    filtered_train_corpus = {doc_id: train_corpus[doc_id] for doc_id in filtered_train_doc_ids}
    
    print(f"Train documents with core classes: {len(filtered_train_doc_ids):,}")
    
    # Prepare test pseudo-labeled documents
    test_pseudo_doc_ids = []
    test_pseudo_core_classes = []
    skipped_multi_level0 = 0
    
    for doc_id, pred in test_pseudo_predictions.items():
        if pred['accepted']:
            selected_classes = pred['selected_classes']
            
            # Filter: If 3 core classes all have different Level 0 ancestors, skip
            if len(selected_classes) == 3:
                level0_ancestors = set()
                for cid in selected_classes:
                    level0 = core_mining.find_level0_ancestor(cid, parents_dict)
                    level0_ancestors.add(level0)
                
                # If all 3 have different Level 0 ancestors, skip (unreliable)
                if len(level0_ancestors) == 3:
                    skipped_multi_level0 += 1
                    continue
            
            doc_id_int = int(doc_id)
            test_pseudo_doc_ids.append(doc_id_int)
            test_pseudo_core_classes.append(selected_classes)
    
    print(f"Accepted test pseudo-labels: {len(test_pseudo_doc_ids):,}")
    print(f"   Skipped (3 classes, all different Level0): {skipped_multi_level0:,}")
    
    # Expand test pseudo-labels
    if len(test_pseudo_doc_ids) > 0:
        test_pseudo_targets, test_pseudo_masks = core_mining.expand_labels(
            test_pseudo_core_classes, parents_dict, children_dict, num_classes
        )
        test_pseudo_corpus = {doc_id: test_corpus[doc_id] for doc_id in test_pseudo_doc_ids}
    
    # Combine datasets
    combined_doc_ids = filtered_train_doc_ids + test_pseudo_doc_ids
    combined_targets = np.concatenate([filtered_train_targets, test_pseudo_targets], axis=0)
    combined_masks = np.concatenate([filtered_train_masks, test_pseudo_masks], axis=0)
    combined_corpus = {**filtered_train_corpus, **test_pseudo_corpus}
    
    print(f"\n✅ Combined Dataset:")
    print(f"   Train (core classes): {len(filtered_train_doc_ids):,}")
    print(f"   Test (pseudo-labeled): {len(test_pseudo_doc_ids):,}")
    print(f"   Total: {len(combined_doc_ids):,}")
    
    # ====== 6. Initialize Fresh Model ======
    print("\n" + "="*80)
    print("Step 6: Initializing Fresh Model")
    print("="*80)
    
    # Re-compute label embeddings (clean start)
    bert_model_temp = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    label_emb_init_fresh = torch.zeros(len(id2class), 768).to(device)
    
    with torch.no_grad():
        for i in range(0, len(class_names), 64):
            batch_names = class_names[i:i+64]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = bert_model_temp(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            label_emb_init_fresh[i:i+64] = embs
    
    del bert_model_temp
    torch.cuda.empty_cache()
    
    # Create fresh model
    fresh_model = models.TaxoClassModel(
        len(id2class), 
        label_emb_init_fresh, 
        adj, 
        model_name="microsoft/deberta-v3-base"
    ).to(device)
    
    print("Initialized fresh model (random weights except label embeddings)")
    
    # ====== 7. Supervised Training ======
    print("\n" + "="*80)
    print("Step 7: Supervised Training (Warm-up Style)")
    print("="*80)
    
    # Training configuration
    epochs = 17
    batch_size = 64
    lr = 5e-5
    checkpoint_dir = "checkpoints_final"
    
    # Custom training loop (to save to checkpoints_final/)
    from torch.utils.data import DataLoader
    
    doc_ids = list(combined_corpus.keys())
    dataset = trainer.TextDataset(doc_ids, combined_corpus, bert_tokenizer, 
                                   targets=combined_targets, masks=combined_masks, max_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=lr)
    
    # Check for existing checkpoints to resume from
    start_epoch = 0
    for i in range(epochs, 0, -1):
        ckpt_path = os.path.join(checkpoint_dir, f'final_epoch_{i}.pth')
        if os.path.exists(ckpt_path):
            print(f"📂 Found existing checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            fresh_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"   Resuming from epoch {start_epoch + 1}")
            break
    
    if start_epoch == 0:
        print("Starting training from scratch...")
    
    for epoch in range(start_epoch, epochs):
        avg_loss = trainer.train_epoch(fresh_model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        
        # Save checkpoint to checkpoints_final/
        ckpt_path = os.path.join(checkpoint_dir, f'final_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': fresh_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    
    print("\n" + "="*80)
    print("✅ Final Training Complete!")
    print("="*80)
    print(f"Checkpoints saved to: checkpoints_final/final_epoch_*.pth")
    print(f"Test pseudo-labels saved to: checkpoints_final/pseudo_labels_test_final.json")


if __name__ == "__main__":
    main()
