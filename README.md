# Hierarchical Multi-Label Text Classification (TaxoClass Implementation)

This repository contains the implementation of a **Hierarchical Multi-Label Text Classification** system for Amazon product reviews. The project is based on the **TaxoClass** framework, which leverages a class taxonomy and label names to classify documents without requiring labeled training data (Zero-shot / Weakly-supervised setting).

## 📌 Project Overview

*   **Task:** Classify unlabeled product reviews into 531 hierarchical categories.
*   **Method:** LLM-Free TaxoClass Framework (Core Class Mining + Self-Training).
*   **Key Features:**
    *   **Core Class Mining:** Generates initial "Silver Labels" using **RoBERTa-MNLI** (Semantic Entailment) and **BM25** (Lexical Matching).
    *   **Hierarchical Label Expansion:** Expands labels to include ancestor nodes (Positive) and masks descendant nodes to handle hierarchy.
    *   **Dual Encoder Model:** Uses **BERT** for document embedding and **GCN (Graph Convolutional Network)** for label embedding.
    *   **Self-Training:** Iteratively refines the model using its own high-confidence predictions.

## 📂 Repository Structure

```
.
├── Amazon_products/       # Dataset directory
│   ├── classes.txt        # List of 531 classes
│   ├── class_hierarchy.txt # Parent-Child relationships
│   ├── class_related_keywords.txt # Keywords for each class
│   ├── train/             # Unlabeled training corpus
│   └── test/              # Test corpus
├── src/                   # Source code
│   ├── core_mining.py     # Silver Label Generation (BM25 + NLI)
│   ├── models.py          # BERT + GCN Model Architecture
│   ├── trainer.py         # Training Loop & Hierarchical Loss
│   └── utils.py           # Data Loading & Graph Construction
├── main.py                # Main execution script
├── pyproject.toml         # Dependency management (uv)
└── README.md              # Project documentation
```

## 🚀 How to Run

### 1. Prerequisites

This project uses `uv` for dependency management. Make sure you have Python 3.9+ installed.

```bash
# Install uv if you haven't already
pip install uv
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Pipeline

The `main.py` script executes the entire pipeline:
1.  Loads data and builds the class hierarchy graph.
2.  Generates Silver Labels using BM25 and RoBERTa-MNLI.
3.  Trains the TaxoClass model (BERT + GCN) with Self-Training.
4.  Predicts on the Test set and saves `submission.csv`.

```bash
uv run main.py
```

## 🛠️ Implementation Details

### Phase 1: Core Class Mining (Silver Label Generation)
*   Calculates **BM25 scores** between documents and class queries (Class Name + Keywords).
*   Calculates **Entailment scores** using `roberta-large-mnli` for Top-K candidates from BM25.
*   Combines scores to select the most probable **Core Class**.

### Phase 2: Hierarchical Label Expansion
*   **Positive (+):** Core Class + All Ancestors.
*   **Masked (?):** All Descendants of the Core Class (excluded from loss calculation).
*   **Negative (-):** All other classes.

### Phase 3: Model Training
*   **Document Encoder:** `bert-base-uncased`
*   **Label Encoder:** 2-Layer GCN with normalized adjacency matrix.
*   **Loss Function:** Hierarchical Binary Cross Entropy (masked).
*   **Self-Training:** Updates Core Classes based on model predictions and retrains.

## 📝 References
*   **TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names** (Shen et al., NAACL 2021)
*   Project Guidelines: `final_project.md`

## 👤 Author
*   **Student ID:** 20252R0136DATA30400
*   **Team Name:** (Your AWS Account ID)
