"""
Self-Training Module for Hierarchical Multi-Label Classification

This module implements self-training strategy to leverage unlabeled data
with Level 0 = 2 property for improving model performance.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from . import core_mining  # Import existing utilities


def identify_unlabeled_documents(core_classes_dict, train_doc_ids):
    """
    Identify documents with NO core classes (empty labels)
    These are the 10% of data excluded from warm-up training.
    
    Args:
        core_classes_dict: Dictionary mapping doc_id -> list of class IDs
        train_doc_ids: List of training document IDs
    
    Returns:
        List of indices where core classes are empty
    """
    unlabeled_indices = []
    
    for i, doc_id in enumerate(train_doc_ids):
        cores = core_classes_dict.get(doc_id, [])
        
        # Find documents with no core classes
        if len(cores) == 0:
            unlabeled_indices.append(i)
    
    return unlabeled_indices


def generate_pseudo_labels(model, dataloader, device, 
                           min_threshold=0.85, 
                           min_num_classes=2,
                           verbose=True):
    """
    Generate pseudo-labels for unlabeled data
    
    Args:
        model: Trained model
        dataloader: DataLoader for unlabeled data
        device: torch device
        min_threshold: Minimum confidence threshold (default: 0.85)
        min_num_classes: Minimum number of classes above threshold (default: 2)
        verbose: Print progress
    
    Returns:
        tuple: (pseudo_targets, pseudo_masks, selected_indices, confidences, all_predictions)
    """
    model.eval()
    
    all_logits = []
    all_doc_ids = []
    
    # Inference
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Generating pseudo-labels") if verbose else dataloader
        
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits.cpu())
            all_doc_ids.extend(batch['doc_id'].tolist())
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(all_logits).numpy()
    
    # Filter by confidence
    pseudo_targets = []
    pseudo_masks = []
    selected_indices = []
    confidences = []
    
    # Store all predictions
    all_predictions = {}
    
    for i in range(len(probs)):
        doc_id = all_doc_ids[i]
        
        # Find classes above threshold
        high_conf_mask = probs[i] > min_threshold
        num_high_conf = high_conf_mask.sum()
        high_conf_indices = np.where(high_conf_mask)[0].tolist()
        
        # Store prediction info
        all_predictions[str(doc_id)] = {
            'selected_classes': high_conf_indices,
            'confidences': probs[i][high_conf_mask].tolist() if num_high_conf > 0 else [],
            'num_classes': int(num_high_conf),
            'accepted': bool(num_high_conf >= min_num_classes)
        }
        
        # Check if at least min_num_classes pass threshold
        if num_high_conf >= min_num_classes:
            # Limit to top-3 classes (hierarchical multi-label constraint)
            if num_high_conf > 3:
                # Get top-3 highest confidence classes
                top3_indices = np.argsort(probs[i])[-3:]
                pseudo_label = np.zeros_like(probs[i], dtype=np.float32)
                pseudo_label[top3_indices] = 1.0
                
                # Update prediction info
                all_predictions[str(doc_id)]['selected_classes'] = top3_indices.tolist()
                all_predictions[str(doc_id)]['confidences'] = probs[i][top3_indices].tolist()
                all_predictions[str(doc_id)]['num_classes'] = 3
                all_predictions[str(doc_id)]['limited_to_top3'] = True
            else:
                # Keep all classes above threshold (already ≤ 3)
                pseudo_label = (probs[i] > min_threshold).astype(np.float32)
                all_predictions[str(doc_id)]['limited_to_top3'] = False
            
            # Create mask (all ones for simplicity)
            mask = np.ones_like(pseudo_label)
            
            pseudo_targets.append(pseudo_label)
            pseudo_masks.append(mask)
            selected_indices.append(i)
            
            # Store average confidence of selected classes
            selected_mask = pseudo_label > 0
            conf = probs[i][selected_mask].mean()
            confidences.append(conf)
    
    pseudo_targets = np.array(pseudo_targets) if pseudo_targets else np.array([])
    pseudo_masks = np.array(pseudo_masks) if pseudo_masks else np.array([])
    
    if verbose:
        print(f"\n📊 Pseudo-labeling Statistics:")
        print(f"   Total candidates: {len(probs):,}")
        print(f"   Pseudo-labeled: {len(selected_indices):,} ({len(selected_indices)/len(probs)*100:.1f}%)")
        if len(confidences) > 0:
            print(f"   Avg confidence: {np.mean(confidences):.3f}")
            print(f"   Min confidence: {np.min(confidences):.3f}")
            print(f"   Max confidence: {np.max(confidences):.3f}")
    
    return pseudo_targets, pseudo_masks, selected_indices, confidences, all_predictions


def combine_datasets(original_doc_ids, original_corpus, original_targets, original_masks,
                    pseudo_doc_ids, pseudo_corpus, pseudo_targets, pseudo_masks):
    """
    Combine original (silver) and pseudo-labeled data
    
    Returns:
        tuple: (combined_doc_ids, combined_corpus, combined_targets, combined_masks)
    """
    # Combine doc IDs
    combined_doc_ids = list(original_doc_ids) + list(pseudo_doc_ids)
    
    # Combine corpus (dictionary)
    combined_corpus = dict(original_corpus)
    combined_corpus.update(pseudo_corpus)
    
    # Combine targets and masks
    combined_targets = np.vstack([original_targets, pseudo_targets])
    combined_masks = np.vstack([original_masks, pseudo_masks])
    
    return combined_doc_ids, combined_corpus, combined_targets, combined_masks


def self_training_iteration(model, tokenizer, device,
                           train_doc_ids, train_corpus, train_targets, train_masks,
                           unlabeled_doc_ids, unlabeled_corpus,
                           parents_dict,
                           children_dict=None,
                           num_classes=None,
                           min_threshold=0.85,
                           min_num_classes=2,
                           batch_size=32,
                           verbose=True):
    """
    Perform one iteration of self-training
    
    Args:
        model: Current model
        tokenizer: BERT tokenizer
        device: torch device
        train_doc_ids, train_corpus, train_targets, train_masks: Original training data
        unlabeled_doc_ids, unlabeled_corpus: Unlabeled data
        parents_dict: Hierarchy information (required)
        children_dict: Children relationships (for expansion)
        num_classes: Total number of classes (for expansion)
        min_threshold: Confidence threshold
        min_num_classes: Minimum classes requirement
        batch_size: Batch size for inference
        verbose: Print progress
    
    Returns:
        tuple: (combined_doc_ids, combined_corpus, combined_targets, combined_masks, stats, all_predictions)
    """
    from . import trainer  # Import here to avoid circular dependency
    from . import core_mining  # For label expansion
    
    print("\n" + "="*80)
    print("Self-Training Iteration")
    print("="*80)
    
    # Create dataloader for unlabeled data
    unlabeled_dataset = trainer.TextDataset(
        unlabeled_doc_ids, 
        unlabeled_corpus, 
        tokenizer
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Generate pseudo-labels (core classes only, top-3 limited)
    pseudo_targets_core, pseudo_masks_core, selected_indices, confidences, all_predictions = generate_pseudo_labels(
        model=model,
        dataloader=unlabeled_dataloader,
        device=device,
        min_threshold=min_threshold,
        min_num_classes=min_num_classes,
        verbose=verbose
    )
    
    # If no pseudo-labels generated, return original data
    if len(selected_indices) == 0:
        print("⚠️  No pseudo-labels generated. Returning original data.")
        return train_doc_ids, train_corpus, train_targets, train_masks, {}, all_predictions
    
    # Select pseudo-labeled documents
    pseudo_doc_ids = [unlabeled_doc_ids[i] for i in selected_indices]
    pseudo_corpus = {did: unlabeled_corpus[did] for did in pseudo_doc_ids}
    
    # --- Expand pseudo-labels to include parents ---
    if children_dict is not None and num_classes is not None:
        print(f"\n🌳 Expanding pseudo-labels hierarchically...")
        
        # Convert pseudo_targets to core class list format for expansion
        pseudo_core_classes = []
        for i in range(len(pseudo_targets_core)):
            core_ids = np.where(pseudo_targets_core[i] > 0)[0].tolist()
            pseudo_core_classes.append(core_ids)
        
        # Expand labels
        pseudo_targets, pseudo_masks = core_mining.expand_labels(
            pseudo_core_classes,
            parents_dict,
            children_dict,
            num_classes
        )
        
        print(f"   Core classes: {pseudo_targets_core.sum():.0f} total")
        print(f"   After expansion: {pseudo_targets.sum():.0f} total (+{(pseudo_targets.sum() - pseudo_targets_core.sum()):.0f} parents)")
    else:
        print(f"\n⚠️  Skipping label expansion (missing children_dict or num_classes)")
        pseudo_targets = pseudo_targets_core
        pseudo_masks = pseudo_masks_core
    
    # Combine datasets
    combined_doc_ids, combined_corpus, combined_targets, combined_masks = combine_datasets(
        train_doc_ids, train_corpus, train_targets, train_masks,
        pseudo_doc_ids, pseudo_corpus, pseudo_targets, pseudo_masks
    )
    
    # Statistics
    stats = {
        'num_original': len(train_doc_ids),
        'num_pseudo': len(pseudo_doc_ids),
        'num_combined': len(combined_doc_ids),
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'selection_rate': len(selected_indices) / len(unlabeled_doc_ids) if unlabeled_doc_ids else 0
    }
    
    if verbose:
        print(f"\n✅ Dataset Combined:")
        print(f"   Original: {stats['num_original']:,}")
        print(f"   Pseudo:   {stats['num_pseudo']:,}")
        print(f"   Total:    {stats['num_combined']:,} (+{stats['num_pseudo']/stats['num_original']*100:.1f}%)")
    
    return combined_doc_ids, combined_corpus, combined_targets, combined_masks, stats, all_predictions
