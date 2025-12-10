"""
Test Corpus Self-Training Module
Iteratively adds high-confidence test documents to training data
"""

import numpy as np
import json
from src import self_training, trainer


def iterative_test_corpus_training(
    model,
    tokenizer,
    device,
    current_labeled_doc_ids,
    current_labeled_corpus,
    current_labeled_targets,
    current_labeled_masks,
    test_corpus,
    parents_dict,
    children_dict,
    num_classes,
    num_iterations=3,
    docs_per_iteration=1500,
    threshold=0.995,
    epochs=3,
    batch_size=64,
    lr=3e-5
):
    """
    Iteratively perform self-training on test corpus
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for encoding
        device: torch device
        current_labeled_doc_ids: Currently labeled document IDs
        current_labeled_corpus: Currently labeled corpus dict
        current_labeled_targets: Current training targets
        current_labeled_masks: Current training masks
        test_corpus: Test corpus dict
        parents_dict: Class hierarchy parents
        children_dict: Class hierarchy children
        num_classes: Total number of classes
        num_iterations: Number of test corpus iterations
        docs_per_iteration: Documents to add per iteration
        threshold: Confidence threshold
        epochs: Training epochs per iteration
        batch_size: Training batch size
        lr: Learning rate
    
    Returns:
        tuple: (model, final_labeled_doc_ids, final_corpus, final_targets, final_masks, used_test_doc_ids)
    """
    
    print("\n" + "="*80)
    print("Iterative Test Corpus Self-Training Phase (Transductive Learning)")
    print("="*80)
    
    # Track already used test documents
    used_test_doc_ids = set()
    
    for test_iter in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Test Corpus Iteration {test_iter + 1}/{num_iterations}")
        print(f"{'='*80}")
        
        # Get remaining test documents
        remaining_test_ids = [did for did in test_corpus.keys() if did not in used_test_doc_ids]
        
        if len(remaining_test_ids) == 0:
            print("\n⚠️  No more test documents remaining. Stopping test self-training.")
            break
        
        print(f"\n📊 Remaining Test Docs: {len(remaining_test_ids):,} / {len(test_corpus):,}")
        print(f"   Already Used: {len(used_test_doc_ids):,}")
        print(f"   Threshold: {threshold}")
        
        remaining_test_corpus = {did: test_corpus[did] for did in remaining_test_ids}
        
        # Perform self-training on remaining test corpus
        combined_doc_ids, combined_corpus, combined_targets, combined_masks, stats, test_pseudo_predictions = \
            self_training.self_training_iteration(
                model=model,
                tokenizer=tokenizer,
                device=device,
                train_doc_ids=current_labeled_doc_ids,
                train_corpus=current_labeled_corpus,
                train_targets=current_labeled_targets,
                train_masks=current_labeled_masks,
                unlabeled_doc_ids=remaining_test_ids,
                unlabeled_corpus=remaining_test_corpus,
                parents_dict=parents_dict,
                children_dict=children_dict,
                num_classes=num_classes,
                min_threshold=threshold,
                min_num_classes=2,
                batch_size=32,
                verbose=False  # Hide verbose output for test corpus (we filter afterward)
            )
        
        # Save test corpus pseudo-label predictions for this iteration
        TEST_PSEUDO_LABELS_PATH = f"checkpoints/pseudo_labels_test_iter{test_iter+1}.json"
        with open(TEST_PSEUDO_LABELS_PATH, 'w') as f:
            json.dump(test_pseudo_predictions, f, indent=2)
        print(f"\n💾 Saved test iteration {test_iter+1} pseudo-labels to {TEST_PSEUDO_LABELS_PATH}")
        
        # Filter to top N highest confidence documents
        if stats.get('num_pseudo', 0) > 0:
            # Extract pseudo-labeled documents
            num_original = len(current_labeled_doc_ids)
            pseudo_doc_ids = combined_doc_ids[num_original:]
            pseudo_targets = combined_targets[num_original:]
            pseudo_masks = combined_masks[num_original:]
            
            # Calculate average confidence for each pseudo-labeled document
            doc_confidences = []
            for i, doc_id in enumerate(pseudo_doc_ids):
                doc_id_str = str(doc_id)
                if doc_id_str in test_pseudo_predictions and test_pseudo_predictions[doc_id_str]['accepted']:
                    avg_conf = np.mean(test_pseudo_predictions[doc_id_str]['confidences'])
                    doc_confidences.append((i, doc_id, avg_conf))
            
            if len(doc_confidences) == 0:
                print(f"\n⚠️  No documents passed threshold in test iteration {test_iter+1}. Stopping.")
                break
            
            # Sort by confidence and select top N
            doc_confidences.sort(key=lambda x: x[2], reverse=True)
            num_to_select = min(docs_per_iteration, len(doc_confidences))
            top_indices = [item[0] for item in doc_confidences[:num_to_select]]
            top_doc_ids = [item[1] for item in doc_confidences[:num_to_select]]
            
            # Filter pseudo data to top N
            filtered_pseudo_targets = pseudo_targets[top_indices]
            filtered_pseudo_masks = pseudo_masks[top_indices]
            filtered_pseudo_corpus = {doc_id: test_corpus[doc_id] for doc_id in top_doc_ids}
            
            # Recombine with current labeled data
            combined_doc_ids = list(current_labeled_doc_ids) + top_doc_ids
            combined_corpus = dict(current_labeled_corpus)
            combined_corpus.update(filtered_pseudo_corpus)
            combined_targets = np.vstack([current_labeled_targets, filtered_pseudo_targets])
            combined_masks = np.vstack([current_labeled_masks, filtered_pseudo_masks])
            
            print(f"\n✅ Pseudo-label Filtering Complete:")
            print(f"   Candidates (threshold {threshold}): {len(doc_confidences):,}")
            print(f"   Selected for training: {num_to_select}")
            print(f"   Confidence range: {doc_confidences[min(num_to_select-1, len(doc_confidences)-1)][2]:.4f} - {doc_confidences[0][2]:.4f}")
            print(f"   New total dataset size: {len(combined_doc_ids):,} (+{num_to_select})")
            
            # Re-train with selected test documents
            print(f"\n{'='*80}")
            print(f"Re-training with Test Data (Iteration {test_iter+1})")
            print(f"{'='*80}")
            
            model = trainer.supervised_training_loop(
                model, combined_corpus, tokenizer,
                combined_targets, combined_masks, device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                checkpoint_prefix=f'retrain_test_iter{test_iter+1}'
            )
            
            # Update for next iteration
            current_labeled_doc_ids = combined_doc_ids
            current_labeled_corpus = combined_corpus
            current_labeled_targets = combined_targets
            current_labeled_masks = combined_masks
            
            # Mark these test docs as used
            used_test_doc_ids.update(top_doc_ids)
            
            print(f"\n✅ Test iteration {test_iter+1} completed!")
            print(f"   Test docs added: +{num_to_select}")
            print(f"   Total labeled: {len(current_labeled_doc_ids):,}")
            print(f"   Total used test docs: {len(used_test_doc_ids):,} / {len(test_corpus):,}")
        else:
            print(f"\n⚠️  No high-confidence test docs in iteration {test_iter+1}. Stopping.")
            break
    
    print(f"\n{'='*80}")
    print(f"Test Corpus Self-Training Complete - {test_iter+1} iterations")
    print(f"Total test docs used: {len(used_test_doc_ids):,} / {len(test_corpus):,}")
    print(f"{'='*80}")
    
    return model, current_labeled_doc_ids, current_labeled_corpus, current_labeled_targets, current_labeled_masks, used_test_doc_ids
