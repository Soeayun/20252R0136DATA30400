# TaxoClass: Hierarchical Multi-Label Text Classification

This repository implements a **Hierarchical Multi-Label Text Classification** system for Amazon product reviews, based on the **TaxoClass** framework with LLM-based refinement.

## 📌 Project Overview

- **Task:** Classify unlabeled product reviews into 531 hierarchical categories
- **Method:** TaxoClass Framework + LLM Refinement + Self-Training
- **Key Components:**
  - **SBERT + Reranker:** Initial candidate retrieval using BGE-M3
  - **LLM Refinement:** GPT-4o-mini for pseudo-label selection
  - **DeBERTa + GAT:** Document encoding with Graph Attention Network for label hierarchy
  - **Self-Training:** Iterative refinement with high-confidence predictions

## 📂 Repository Structure

```
.
├── Amazon_products/           # Dataset directory
│   ├── classes.txt            # 531 class names
│   ├── class_hierarchy.txt    # Parent-Child edges
│   ├── class_related_keywords.txt
│   ├── train/                 # Training corpus
│   └── test/                  # Test corpus
├── src/                       # Source code
│   ├── core_mining.py         # SBERT + Reranker + Candidate filtering
│   ├── models.py              # DeBERTa + GAT Model Architecture
│   ├── trainer.py             # Training Loop
│   ├── llm_refinement.py      # LLM-based pseudo labeling
│   ├── self_training.py       # Self-training iteration
│   └── utils.py               # Data Loading & Graph Construction
├── checkpoints/               # Model checkpoints & cached files
├── main.py                    # Main training script
├── predict_submission.py      # Prediction script
├── run_llm_refinement.py      # LLM refinement script
├── run_pipeline.sh            # Full pipeline script
├── pyproject.toml             # Dependencies (uv)
└── README.md
```

## 🚀 Quick Start (Reproducibility)

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- `uv` package manager

```bash
# Install uv if not already installed
pip install uv
```

### Run the Full Pipeline

```bash
# Clone the repository
git clone https://github.com/Soeayun/20252R0136DATA30400.git
cd 20252R0136DATA30400

# Install dependencies
uv sync

# Set up OpenAI API key (REQUIRED for LLM refinement)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Run the complete pipeline
uv run main.py

# Generate final predictions
uv run predict_submission.py
```

### What `main.py` Does Automatically

The script handles all stages with caching:

1. **Step 4.1**: Generate doc candidates (SBERT + Reranker) → `doc_candidates.json`
2. **Step 4.2**: Identify core classes → `core_classes.json`
3. **Step 4.3**: LLM refinement (**REQUIRED**) → `core_classes_llm_refined.json`
4. **Step 4.4**: Post-processing (Level 0 limit)
5. **Step 5+**: Model training with self-training

Each step checks for cached files and skips if already computed.

**⚠️ Note**: The OPENAI_API_KEY is **required** for LLM refinement. The script will exit with an error if the API key is not found and `core_classes_llm_refined.json` doesn't exist.

### Training from Scratch

If you want to train the model from scratch (without using pre-computed checkpoints), delete all files in the `checkpoints/` directory:

```bash
rm -rf checkpoints/*
```

Then run `uv run main.py` to start the full pipeline from the beginning.

## ⚙️ Configuration

### LLM Refinement Setup

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```


## 🛠️ Implementation Details

### Phase 1: Core Class Mining

1. **SBERT Retrieval** (BGE-M3)
   - Encode documents and class hierarchy paths
   - Retrieve Top-100 candidates per document

2. **Cross-Encoder Reranking** (BGE-Reranker-v2-M3)
   - Re-score candidates with full context
   - Filter by score threshold (>0.5)

3. **Candidate Filtering**
   - Keep Top-15 candidates per document
   - All candidates sent to LLM for final selection

### Phase 2: LLM Refinement

- **Model:** GPT-4o-mini
- **Task:** Select exactly ONE category from hierarchy paths, or NONE
- **Prompt:** Shows full hierarchy path (e.g., "beauty > hair care > styling tools")
- **Output:** Single class ID or -1 (none), ancestors auto-included

### Phase 3: Model Training

- **Document Encoder:** DeBERTa-v3-base (768-dim)
- **Label Encoder:** 2-Layer GAT (4 heads, skip connection)
- **Training:** 
  - Supervised warm-up with silver labels (40 epochs)
  - Self-training with pseudo-labels (2 iterations)
- **Loss:** Binary Cross-Entropy with Logits

### Phase 4: Prediction

- **Thresholding:** Classes with probability > 0.65
- **Constraints:** 2-3 labels per document
- **Output:** `submission.csv`


## 📝 References

- **TaxoClass:** Shen et al., "Hierarchical Multi-Label Text Classification Using Only Class Names" (NAACL 2021)
- **BGE-M3:** BAAI General Embedding
- **DeBERTa:** Microsoft DeBERTa-v3


