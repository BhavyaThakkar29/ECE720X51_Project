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

## Datasets

All 6 datasets are streamed directly from HuggingFace — nothing is downloaded locally. They are all called from `src/evaluation/dataset_loader.py`. Each dataset has its own function in that file. Below is a description of each dataset, which similarity level it is used for, and exactly how to call it.

Install the library first if you have not already:

```bash
pip install datasets
```

---

### Word-level Datasets

These are used to evaluate **S_word** — token-level cosine similarity between individual word embeddings.

#### SimpleQA
A short-form factual QA dataset released by OpenAI with 4,326 questions covering geography, history, science, and pop culture. Answers are single, unambiguous correct answers — typically just a few words long. This makes it ideal for word-level similarity since the answers are short and precise, so the embedder must capture exact lexical meaning rather than relying on sentence context.

HuggingFace page: https://huggingface.co/datasets/basicv8vc/SimpleQA

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_simpleqa(n=500):
    ds = load_dataset("basicv8vc/SimpleQA", split="test", streaming=True)
    records = []
    for row in ds.take(n):
        ref = row["answer"]
        records.append({"generated": ref, "reference": ref, "label": 0})
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_simpleqa

records = load_simpleqa(n=500)
result = evaluator.run(records, dataset_name="simpleqa")
```

---

#### BioASQ
A biomedical question answering dataset sourced from PubMed literature. Answers are short medical terms, gene names, drug names, or brief factual phrases. The highly specialised vocabulary makes it a strong stress test for word-level embeddings — the embedder has to capture domain-specific semantic meaning and cannot rely on common word overlap. Requires accepting dataset terms on HuggingFace before streaming will work — visit https://huggingface.co/datasets/bigbio/bioasq_task_b and click Agree while logged in.

HuggingFace page: https://huggingface.co/datasets/bigbio/bioasq_task_b

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_bioasq(n=500):
    ds = load_dataset(
        "bigbio/bioasq_task_b",
        name="bioasq_task_b_source",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    records = []
    for row in ds.take(n):
        if not row.get("ideal_answer"):
            continue
        ref = row["ideal_answer"][0] if isinstance(row["ideal_answer"], list) else row["ideal_answer"]
        records.append({"generated": ref, "reference": ref, "label": 0})
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_bioasq

records = load_bioasq(n=500)
result = evaluator.run(records, dataset_name="bioasq")
```

---

### Sentence-level Datasets

These are used to evaluate **S_sent** — cosine similarity computed between sentence-level embeddings.

#### TriviaQA
A large-scale reading comprehension and QA dataset with over 650,000 question-answer-evidence triples. Questions are trivia-style and answers are typically one to two full sentences pulled from Wikipedia or web documents. Useful for sentence-level evaluation because answers are full sentences rather than single words, which tests whether the sentence embedder captures meaning at the clause level.

HuggingFace page: https://huggingface.co/datasets/mandarjoshi/trivia_qa

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_triviaqa(n=500):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
    records = []
    for row in ds.take(n):
        ref = row["answer"]["value"]
        records.append({"generated": ref, "reference": ref, "label": 0})
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_triviaqa

records = load_triviaqa(n=500)
result = evaluator.run(records, dataset_name="triviaqa")
```

---

#### SummEval
A summarization evaluation dataset containing machine-generated summaries from 16 different summarization models, each rated by human annotators across four dimensions: consistency, coherence, fluency, and relevance. The consistency scores directly map to faithfulness, making it ideal for validating S_sent — you can check whether your sentence-level cosine similarity correlates with the human consistency ratings rather than just running a binary label.

HuggingFace page: https://huggingface.co/datasets/mteb/summeval

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_summeval(n=500):
    ds = load_dataset("mteb/summeval", split="test", streaming=True)
    records = []
    for row in ds.take(n):
        machine_summary = row["machine_summaries"][0] if row.get("machine_summaries") else ""
        human_summary   = row["human_summaries"][0]   if row.get("human_summaries")  else ""
        consistency     = row["human_scores"]["consistency"]
        avg             = sum(consistency) / len(consistency) if consistency else 3
        label           = 0 if avg >= 3 else 1
        records.append({"generated": machine_summary, "reference": human_summary, "label": label})
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_summeval

records = load_summeval(n=500)
result = evaluator.run(records, dataset_name="summeval")
```

---

### Document-level Datasets

These are used to evaluate **S_doc** — cosine similarity computed between single full-document embeddings.

#### TruthfulQA
A benchmark of 817 questions designed to test whether language models produce truthful answers. Questions span health, law, finance, and common misconceptions — areas where models frequently hallucinate plausible-sounding but false information. Each question includes a correct best answer and a set of incorrect answers, giving you ready-made faithful and hallucinated pairs for document-level evaluation. Small dataset but high quality labels.

HuggingFace page: https://huggingface.co/datasets/truthfulqa/truthful_qa

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_truthfulqa(n=500):
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", streaming=True)
    records = []
    for row in ds.take(n):
        correct   = row["best_answer"]
        incorrect = row["incorrect_answers"][0] if row["incorrect_answers"] else ""
        records.append({"generated": correct,   "reference": correct, "label": 0})
        records.append({"generated": incorrect, "reference": correct, "label": 1})
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_truthfulqa

records = load_truthfulqa(n=500)
result = evaluator.run(records, dataset_name="truthfulqa")
```

---

#### QASPER
A dataset of 5,049 questions over 1,585 NLP research papers where answers are extracted from or abstractively generated from the full paper text. Because the reference is an entire research paper, this is the most demanding test for document-level cosine similarity — the embedder must capture faithfulness across long, dense technical documents rather than short paragraphs. Also directly relevant to the project since the domain is NLP research.

HuggingFace page: https://huggingface.co/datasets/allenai/qasper

How it is called in `dataset_loader.py`:
```python
from datasets import load_dataset

def load_qasper(n=500):
    ds = load_dataset("allenai/qasper", split="validation", streaming=True)
    records = []
    for row in ds.take(n):
        for answers in row.get("qas", {}).get("answers", []):
            for ans in answers.get("answer", []):
                free_form = ans.get("free_form_answer", "")
                if not free_form:
                    continue
                records.append({"generated": free_form, "reference": free_form, "label": 0})
                if len(records) >= n:
                    return records
    return records
```

How to use it in `main.py`:
```python
from src.evaluation.dataset_loader import load_qasper

records = load_qasper(n=500)
result = evaluator.run(records, dataset_name="qasper")
```

---
