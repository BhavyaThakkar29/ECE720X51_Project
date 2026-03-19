# Multi-Granularity Cosine Similarity for Hallucination Evaluation

A hallucination evaluation framework that computes cosine similarity at **word**, **sentence**, and **document** levels, then combines them into a single unified metric:

```
S_final = α·S_word + β·S_sent + γ·S_doc   (α + β + γ = 1)
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/yourname/mgcs_hallucination.git
cd mgcs_hallucination
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run evaluation with default config
python main.py --config configs/config.yaml

# 3. Run on a specific dataset with custom weights
python main.py --dataset halueval_qa --alpha 0.3 --beta 0.4 --gamma 0.3
```

---

## File & Folder Descriptions

### Root Files

| File | What it does |
|---|---|
| `main.py` | Entry point. Reads the config, loads a dataset, runs the full MGCS pipeline, and saves results to `results/`. Run this to start an experiment. |
| `requirements.txt` | All Python dependencies (PyTorch, Transformers, Sentence-Transformers, NLTK, scikit-learn, etc.). Run `pip install -r requirements.txt` once after cloning. |
| `README.md` | This file. Project overview and guide for the team. |

---

### `configs/`

| File | What it does |
|---|---|
| `config.yaml` | Central settings file. Controls the three weights (α, β, γ), which embedding models to use at each level, which datasets to evaluate on, batch size, and output directory. Change values here instead of editing source code. |

---

### `src/embeddings/`
Responsible for turning raw text into numerical vectors at each granularity level.

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package and exposes the three embedder classes for easy importing. |
| `word_embedder.py` | Uses a BERT model to extract token-level embeddings. Handles subword tokens by averaging them back into whole-word vectors. Returns one embedding vector per word in the input text. |
| `sentence_embedder.py` | Uses Sentence-Transformers to embed text at the sentence level. First splits the input into individual sentences using NLTK, then encodes each sentence separately. Returns one embedding vector per sentence. |
| `document_embedder.py` | Uses Sentence-Transformers to encode the entire input text as a single fixed-size vector. Treats the whole document as one unit regardless of length. |

---

### `src/similarity/`
Takes embeddings from the embedders above and computes cosine similarity scores.

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package and exposes the three similarity classes. |
| `word_similarity.py` | Computes S_word. Builds a pairwise cosine similarity matrix between all word embeddings in the generated text and the reference text. For each generated word, finds the most similar word in the reference (soft alignment), then averages those scores. |
| `sentence_similarity.py` | Computes S_sent. Does the same soft-alignment approach as word similarity but at the sentence level — matches each generated sentence to its closest reference sentence and averages the results. |
| `document_similarity.py` | Computes S_doc. The simplest of the three — directly computes cosine similarity between the single document embedding of the generated text and the single document embedding of the reference text. Returns one scalar. |

---

### `src/aggregation/`

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package. |
| `aggregator.py` | The core of the project. Takes a generated text and a reference text, calls all three similarity modules, and combines their scores using the weighted formula S_final = α·S_word + β·S_sent + γ·S_doc. Returns an `MGCSScores` dataclass containing all four scores (word, sentence, document, and final) for inspection. |

---

### `src/evaluation/`
Handles loading datasets and measuring how well MGCS detects hallucinations.

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package. |
| `halueval_loader.py` | Downloads and parses the HaluEval dataset from HuggingFace. Supports three splits: `qa`, `summarization`, and `dialogue`. Returns a list of records, each containing a generated text, a reference text, and a binary label (0 = faithful, 1 = hallucinated). |
| `evaluator.py` | Runs the full evaluation loop. Takes a list of records, scores each one using the aggregator, then calls the metrics module to report results. Saves per-sample scores and a summary JSON to `results/`. |
| `metrics.py` | Calculates evaluation metrics given MGCS scores and ground-truth labels. Reports AUROC, AUPRC, Accuracy, Precision, Recall, and F1 for hallucination detection. Also includes a helper to compute mean scores across a dataset. |

---

### `src/utils/`
Shared helper functions used across the project.

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package. |
| `preprocessing.py` | Text cleaning utilities — normalises unicode characters, collapses extra whitespace, and can truncate texts that are too long before embedding. |
| `logger.py` | Sets up a consistent logger that prints timestamped messages to the console. Import and use in any module to keep log formatting uniform across the project. |

---

### `data/`

| Folder | What it does |
|---|---|
| `raw/` | Store original downloaded datasets here, untouched. Treat as read-only so there is always a clean backup. |
| `processed/` | Store cleaned and tokenized versions of the datasets here, after running any preprocessing scripts. |
| `compiled/` | **Start here.** This is where the final, merged, labelled datasets live (e.g. `halueval_qa.json`, `stsb.json`). Every team member must use the same files from this folder to ensure consistent experiments. |

---

### `tests/`
Unit tests to verify each module works correctly in isolation, using mock embeddings so no models need to be downloaded to run them.

| File | What it does |
|---|---|
| `__init__.py` | Marks this folder as a Python package. |
| `test_aggregator.py` | Tests the aggregator — verifies the weighted formula is applied correctly, that weights must sum to 1, and that identical texts return a score of 1.0. |
| `test_similarity.py` | Tests all three similarity modules — checks that identical embeddings return 1.0, orthogonal embeddings return 0.0, and edge cases like empty inputs are handled gracefully. |
| `test_metrics.py` | Tests the metrics module — verifies perfect separation returns AUROC of 1.0 and that summarize_scores correctly averages across samples. |

Run all tests with: `pytest tests/`

---

### `notebooks/`

| File | What it does |
|---|---|
| `exploratory_analysis.ipynb` | Interactive notebook for analysis and visualisation. Loads a small HaluEval sample, computes MGCS scores, plots score distributions for faithful vs. hallucinated texts, draws ROC curves comparing S_word/S_sent/S_doc/S_final, and runs a grid search over weight combinations to find the best α, β, γ. |

---

### `results/`
Empty at the start. After running `main.py`, evaluation outputs are saved here automatically — one JSON file with per-sample scores and one summary JSON with aggregate metrics per dataset.

---

## Benchmark Datasets

| Granularity | Dataset | Task |
|---|---|---|
| Word | WordSim-353, SimLex-999 | Lexical similarity |
| Sentence | STS-B, SNLI, HaluEval QA | Sentence faithfulness |
| Document | HaluEval Summarization, CNN/DM | Full-text grounding |

---

## Configuration Reference

```yaml
weights:
  alpha: 0.3   # word-level weight  (S_word)
  beta: 0.4    # sentence-level weight (S_sent)
  gamma: 0.3   # document-level weight (S_doc)

models:
  word: "bert-base-uncased"
  sentence: "all-MiniLM-L6-v2"
  document: "all-mpnet-base-v2"
```
