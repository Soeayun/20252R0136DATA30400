"""
Test Corpus Self-Training using TaxoClass's original KL-divergence based approach
Soft selection with dynamic Q distribution (sharpened targets)
"""

from src import trainer


def taxoclass_test_corpus_training(
    model,
    tokenizer,
    device,
    current_labeled_corpus,
    test_corpus,
    num_iterations=5,   # More iterations for frequent Q updates
    epochs_per_iter=1,  # Fewer epochs to prevent stale Q
    batch_size=32,
    lr=1e-5
):
    """
    Apply TaxoClass's original Self-training (KL divergence) on test corpus
    
    Uses the entire test corpus with soft selection:
    - Q distribution = Sharpened P (high confidence pushed higher, low pushed lower)
    - Loss = KL(Q||P)
    - No hard threshold, automatically down-weights uncertain predictions
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        device: torch device
        current_labeled_corpus: Current labeled training data
        test_corpus: Test corpus (unlabeled)
        num_iterations: Number of self-training iterations
        epochs_per_iter: Training epochs per iteration
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        model: Updated model
    """
    
    print("\n" + "="*80)
    print("TaxoClass Self-Training on Test Corpus (Soft Selection)")
    print("="*80)
    print(f"\nMethod: KL Divergence with Dynamic Q Distribution")
    print(f"- Uses ALL test documents: {len(test_corpus):,}")
    print(f"- Soft selection: High-confidence auto-weighted, low-confidence down-weighted")
    print(f"- No hard threshold needed")
    print(f"- Iterations: {num_iterations}, Epochs per iteration: {epochs_per_iter}")
    
    # Call the existing self_training_loop from trainer.py
    # Note: We don't need initial_targets/masks for test corpus
    # The loop will compute Q dynamically from predictions
    
    
    model = trainer.self_training_loop(
        model=model,
        train_corpus=current_labeled_corpus,  # Use as base (won't be updated, just for context)
        test_corpus=test_corpus,              # Test corpus to learn from
        tokenizer=tokenizer,
        initial_targets=None,  # Not used in this context
        initial_masks=None,    # Not used in this context
        parents_dict=None,     # Not needed for KL-based self-training
        children_dict=None,    # Not needed for KL-based self-training
        num_classes=None,      # Will be inferred from model
        device=device,
        num_iterations=num_iterations,
        epochs_per_iter=epochs_per_iter,
        batch_size=batch_size,
        lr=lr
    )
    
    print(f"\n✅ TaxoClass Self-Training Complete")
    print(f"   All {len(test_corpus):,} test documents processed")
    print("="*80)
    
    return model
