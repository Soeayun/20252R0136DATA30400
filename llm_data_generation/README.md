# LLM Synthetic Review Generation & Training Pipeline

Complete pipeline for generating synthetic reviews and training a hierarchical classifier.

## Files

1. **1_select_classes.py** - Select classes for review generation
2. **2_generate_reviews.py** - Generate synthetic reviews using GPT-4o-mini
3. **3_analyze_distribution.py** - Analyze generated review distribution
4. **4_train_model.py** - Train DeBERTa with hierarchical labels
5. **5_predict.py** - Predict on test set

## Quick Start

### Step 1: Generate Synthetic Reviews (Optional - Already Done)

```bash
cd llm_data_generation

# Select classes
python 1_select_classes.py

# Generate reviews (~40 mins, $3-5)
python 2_generate_reviews.py

# Analyze distribution
python 3_analyze_distribution.py
```

### Step 2: Train Model

```bash
# Train DeBERTa with hierarchical labels
python 4_train_model.py

# Expected output:
# - best_model.pt (saved model)
# - Training logs with F1 scores
```

**Training Details:**
- Model: DeBERTa-v3-base
- Hierarchical labels: Each review labeled with [class, parent, grandparent, ...]
- Class embeddings: 128-dim learned representations
- Epochs: 10
- Batch size: 16

### Step 3: Generate Predictions

```bash
# Predict on test set
python 5_predict.py

# Output: predictions.csv
```

## Generated Data

**Current Status (from generation_metadata.json):**
- Total reviews: 6,783
- API calls: 984 (1000 limit)
- Avg reviews per call: ~7 (target was 15, limited by MAX_TOKENS)
- Total cost: ~$0.42

**Distribution:**
- Level 1: 559 reviews (64 classes)
- Level 2: 6,256 reviews (462 classes)
- Quality score: 41.5% (limited by token truncation)

## Model Architecture

```
Input Review Text
    ↓
DeBERTa Encoder → [CLS] embedding (768-dim)
    ↓
Class Embeddings (532 classes × 128-dim)
    ↓
Concat [text_embed, class_embed]
    ↓
MLP Classifier → Binary prediction per class
    ↓
Hierarchical Labels (include all ancestors)
```

## Key Features

1. **Hierarchical Label Expansion**
   - Category 123 → [123, parent, grandparent, ...]
   - Ensures model learns hierarchy naturally

2. **Class Embeddings**
   - Learn 128-dim representation for each class
   - Helps model differentiate similar classes

3. **Multi-label Classification**
   - BCEWithLogitsLoss for independent class probabilities
   - Threshold-based (0.5) for predictions

## Requirements

```bash
pip install transformers torch pandas numpy tqdm scikit-learn openai python-dotenv
```

## Notes

- Token limit issue: MAX_TOKENS=2500 caused truncation → avg 7 reviews/call instead of 15
- To fix: Set MAX_TOKENS=4000 in 2_generate_reviews.py
- Current data is sufficient for training (6,800 reviews)
- Hierarchical labels ensure all classes get some signal through parent relationships
