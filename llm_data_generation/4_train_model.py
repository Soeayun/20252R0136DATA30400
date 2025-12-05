"""
Step 4: Train TaxoClass model on LLM-generated synthetic reviews

Uses the exact same pipeline as main.py but with synthetic review data:
- Data: LLM-generated reviews with hierarchical labels
- Model: TaxoClassModel (DeBERTa + GNN)
- Training: Supervised warmup (same as main.py)
- Label Expansion: core_mining.expand_labels()

Output: Trained model for prediction
"""

import os
import sys
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel

sys.path.append('..')
from src import utils, core_mining, models, trainer

def load_synthetic_data(id2class):
    """
    Load synthetic reviews and convert to core_classes format
    
    Args:
        id2class: Valid class ID mapping for validation
    """
    print("Loading synthetic reviews...")
    
    # Load reviews
    reviews = []
    with open('synthetic_reviews.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    
                    # Handle nested structure
                    if 'product_reviews' in data:
                        # Nested: {"product_reviews": [{...}, {...}]}
                        reviews.extend(data['product_reviews'])
                    elif 'category_id' in data and 'review' in data:
                        # Direct: {"category_id": X, "review": "..."}
                        reviews.append(data)
                    # Skip malformed entries
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(reviews)} synthetic reviews")
    
    # Create corpus dict {doc_id: text}
    # Filter out invalid category IDs
    corpus = {}
    doc_to_category = {}
    skipped = 0
    
    for i, review in enumerate(reviews):
        if 'category_id' not in review or 'review' not in review:
            continue  # Skip incomplete reviews
        
        category_id = review['category_id']
        
        # Validate category ID
        if category_id not in id2class:
            skipped += 1
            continue
        
        doc_id = str(len(corpus))  # Sequential IDs for valid reviews only
        corpus[doc_id] = review['review']
        doc_to_category[doc_id] = category_id
    
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} reviews with invalid category IDs")
    
    print(f"Created corpus with {len(corpus)} valid reviews")
    return corpus, doc_to_category


def build_core_classes(doc_to_category, doc_ids):
    """
    Convert category IDs to core_classes format
    
    Args:
        doc_to_category: {doc_id: category_id}
        doc_ids: List of doc IDs in order
    
    Returns:
        core_classes: List of lists [[class_id], [class_id], ...]
    """
    core_classes = []
    for doc_id in doc_ids:
        category_id = doc_to_category.get(doc_id, None)
        if category_id is not None:
            core_classes.append([category_id])  # Single class per review
        else:
            core_classes.append([])
    
    return core_classes


def main():
    print("="*70)
    print("TAXOCLASS TRAINING ON SYNTHETIC REVIEWS")
    print("="*70)
    
    # Setup
    utils.set_seed(42)
    device = utils.get_device()
    print(f"\nUsing device: {device}")
    
    # Paths
    DATA_DIR = "../Amazon_products"
    CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
    HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
    
    # Load class info
    print("\nLoading class hierarchy...")
    id2class, class2id = utils.load_classes(CLASSES_PATH)
    edges = utils.load_hierarchy(HIERARCHY_PATH)
    num_classes = len(id2class)
    print(f"Total classes: {num_classes}")
    
    # Build hierarchy relations
    adj = utils.build_adjacency_matrix(num_classes, edges).to(device)
    parents_dict, children_dict = utils.get_hierarchy_relations(edges, num_classes)
    
    # Load synthetic data
    corpus, doc_to_category = load_synthetic_data(id2class)
    doc_ids = sorted(list(corpus.keys()))
    
    # Build core_classes (ground truth from LLM)
    print("\nBuilding core classes from synthetic labels...")
    core_classes = build_core_classes(doc_to_category, doc_ids)
    
    # Verify
    labeled_docs = sum(1 for cls_list in core_classes if len(cls_list) > 0)
    print(f"Labeled documents: {labeled_docs}/{len(doc_ids)}")
    
    # Label Expansion (include all ancestors)
    print("\nExpanding labels hierarchically (Limited)...")
    import custom_mining
    targets, masks = custom_mining.expand_labels_limited(
        core_classes, parents_dict, children_dict, num_classes
    )
    
    # Verify expansion
    avg_labels_before = sum(len(cls_list) for cls_list in core_classes) / len(core_classes)
    
    # Convert to tensor if needed for calculation
    if isinstance(targets, torch.Tensor):
        avg_labels_after = targets.sum(dim=1).float().mean().item()
    else:
        # numpy array
        avg_labels_after = targets.sum(axis=1).mean()
    
    print(f"Avg labels per doc: {avg_labels_before:.2f} → {avg_labels_after:.2f} (after expansion)")
    
    # Initialize Model (same as main.py)
    print("\nInitializing TaxoClass model...")
    print("Computing initial label embeddings...")
    
    bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    
    label_emb_init = torch.zeros(num_classes, 768).to(device)
    
    # Batch process label embeddings
    class_names = [id2class[i] for i in range(num_classes)]
    batch_size_emb = 64
    
    with torch.no_grad():
        for i in range(0, len(class_names), batch_size_emb):
            batch_names = class_names[i:i+batch_size_emb]
            inputs = bert_tokenizer(batch_names, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = bert_model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            label_emb_init[i:i+batch_size_emb] = embs
    
    # Create TaxoClass model
    model = models.TaxoClassModel(
        num_classes, label_emb_init, adj, 
        model_name="microsoft/deberta-v3-base"
    ).to(device)
    
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Supervised Training (same as main.py warmup)
    print("\n" + "="*70)
    print("SUPERVISED TRAINING")
    print("="*70)
    
    model = trainer.supervised_training_loop(
        model, corpus, bert_tokenizer,
        targets, masks, device,
        epochs=14,
        batch_size=64,
        lr=5e-5
    )
    
    # Save model
    MODEL_PATH = "taxoclass_synthetic.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✓ Saved model to {MODEL_PATH}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
