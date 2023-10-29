# Quora Question Pairs — Duplicate Detection

> **Notebook:** [`quora_question_pairs_optimize.ipynb`](quora_question_pairs_optimize.ipynb)
> Built for scale — every stage from text cleaning to model training is GPU-accelerated or parallelized.

---

## Table of Contents

1. [Overview](#overview)
2. [Performance Optimizations](#performance-optimizations)
3. [Dataset](#dataset)
4. [Setup & Installation](#setup--installation)
5. [Pipeline](#pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Modeling & Results](#modeling--results)
8. [Saved Artifacts](#saved-artifacts)
9. [Inference](#inference)
10. [Requirements](#requirements)
11. [Contributing](#contributing)
12. [Authors](#authors)

---

## Overview

With Quora receiving over 100 million monthly visitors, users frequently post semantically equivalent questions phrased differently. This project applies NLP and machine learning to classify whether a given pair of questions are duplicates — reducing content redundancy and improving information retrieval.

The pipeline covers end-to-end: GPU-accelerated text preprocessing, hand-crafted NLP features, pre-trained Word2Vec embeddings, Bayesian hyperparameter tuning, and a serialized inference script for production use.

---

## Performance Optimizations

The notebook (`quora_question_pairs_optimize.ipynb`) is purpose-built for speed on large data. Every bottleneck has been addressed:

| Stage | Technique | Speedup |
|-------|-----------|---------|
| Text Preprocessing | RAPIDS `cudf` — all regex, lowercasing, contraction expansion on GPU | Orders of magnitude vs. CPU pandas |
| Length Feature Engineering | GPU DataFrame ops via `cudf`, no Python loops | Vectorized, near-instant |
| Token Feature Extraction | `joblib.Parallel(prefer='threads')` across all CPU cores | ~9,000 rows/sec on 404k rows |
| Fuzzy String Matching | `rapidfuzz` (GIL-free) + chunked round-robin threading via `joblib` | Eliminates per-row Python overhead |
| Word2Vec Embedding | `multiprocessing.Pool` (one process per CPU core) | ~37 sec per question column on 404k rows |
| Hyperparameter Tuning | Optuna TPE + `MedianPruner` + `eval_set` early stopping | Finds good params in 15 trials vs. exhaustive grid |
| Model Training | XGBoost (`cuda`), LightGBM (`gpu`), CatBoost (`GPU`) | Full GPU utilisation on 282k × 623 feature matrix |

---

## Dataset

**Source:** [Kaggle — Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs)

| Property | Value |
|----------|-------|
| Raw rows | 404,290 |
| Usable rows (after null drop) | 404,287 |
| Columns | 6 |
| Target | `is_duplicate` (binary) |

**Fields:**

| Column | Description |
|--------|-------------|
| `id` | Unique pair identifier |
| `qid1`, `qid2` | Individual question identifiers |
| `question1`, `question2` | Raw question text |
| `is_duplicate` | Label — `1` if duplicate, `0` otherwise |

---

## Setup & Installation

### 1. Download the Dataset

```bash
kaggle competitions download -c quora-question-pairs -p /content/
unzip /content/train.csv.zip
```

### 2. Install Dependencies

```bash
pip install numpy==2.0.2 pandas==2.2.2 scikit-learn==1.6.1 \
    xgboost==3.2.0 lightgbm==4.6.0 catboost==1.2.10 \
    optuna==4.8.0 gensim==4.4.0 rapidfuzz==3.14.5 \
    fuzzywuzzy==0.18.0 Distance==0.1.3 nltk==3.9.1 \
    hvplot==0.12.2 holoviews==1.22.1 bokeh==3.8.2 \
    tqdm==4.67.3 joblib==1.5.3 matplotlib==3.10.0 seaborn==0.13.2

# RAPIDS GPU support — requires CUDA 12
pip install cudf-cu12==26.2.1 cupy-cuda12x==14.0.1
```

> Word2Vec embeddings (`word2vec-google-news-300`, 1.6 GB) are downloaded automatically via `gensim.downloader` on first run and cached locally — no manual download needed.

---

## Pipeline

```
Raw CSV
  └── EDA (shape, nulls, class distribution)
        └── GPU Preprocessing (cudf)
              └── Feature Engineering
                    ├── Length features       (cudf, GPU)
                    ├── Token features        (joblib, CPU parallel)
                    ├── Fuzzy features        (rapidfuzz, CPU parallel)
                    └── Word2Vec embeddings   (multiprocessing, CPU parallel)
                          └── Train / Valid / Test Split (70 / 15 / 15)
                                └── StandardScaler
                                      └── Optuna Tuning + GPU Model Training
                                            └── Evaluation → Save Artifacts
```

### Exploratory Data Analysis

- 404,290 rows; 3 nulls dropped
- Class distribution visualized (duplicate vs. non-duplicate bar chart)
- Per-question frequency distribution (log-scale histogram)

### Preprocessing

All text cleaning runs on GPU via `cudf.Series.str` operations:

- Lowercase and strip whitespace
- Remove emojis (Unicode block ranges)
- Strip HTML tags and URLs
- Expand symbols: `%` → `percent`, `$` → `dollar`, `₹` → `rupee`, etc.
- Expand 70+ English contractions (`can't` → `cannot`, etc.)
- Remove punctuation and collapse extra whitespace

---

## Feature Engineering

The final feature matrix has **623 columns** per row.

### Length Features — 8 columns *(GPU, `cudf`)*

| Feature | Description |
|---------|-------------|
| `q1_char_len`, `q2_char_len` | Character length of each question |
| `q1_token_no`, `q2_token_no` | Word count of each question |
| `abs_len_diff` | Absolute difference in word counts |
| `mean_len` | Average word count across both questions |
| `min_char_len`, `max_char_len` | Min and max character lengths |

### Token Features — 11 columns *(CPU parallel, `joblib`)*

| Feature | Description |
|---------|-------------|
| `common_words` | Number of shared tokens |
| `total_words` | Combined token count |
| `ratio_of_common_to_total` | Common / total token ratio |
| `cwc_min`, `cwc_max` | Common non-stopword ratio over min/max stopword sets |
| `csc_min`, `csc_max` | Common non-stopword ratio over min/max token sets |
| `ctc_min`, `ctc_max` | Common token ratio over min/max token sets |
| `first_word_eq`, `last_word_eq` | Boolean first/last word match |

### Fuzzy Matching Features — 4 columns *(CPU parallel, `rapidfuzz`)*

| Feature | Description |
|---------|-------------|
| `fuzz_ratio` | Character-level edit similarity |
| `fuzz_partial_ratio` | Best partial substring match |
| `token_sort_ratio` | Similarity after alphabetically sorting tokens |
| `token_set_ratio` | Set-based token similarity |

### Word2Vec Embedding Features — 600 columns *(CPU parallel, `multiprocessing`)*

- Model: `word2vec-google-news-300` (300-dimensional vectors)
- Each question represented as the mean of its token vectors
- Produces `question1_embedding_0` … `question1_embedding_299` and equivalent for `question2`

---

## Modeling & Results

### Data Split

| Split | Rows | Share |
|-------|------|-------|
| Train | ~282,000 | 70% |
| Validation | ~61,000 | 15% |
| Test | ~61,000 | 15% |

All features standardized with `StandardScaler` before training.

### Hyperparameter Tuning

Bayesian optimization via **Optuna** — 15 trials per model using Tree-structured Parzen Estimators (TPE). `MedianPruner` terminates trials that are unlikely to improve on the median. XGBoost and LightGBM additionally use `eval_set` early stopping to avoid overfitting within each trial.

### Results

| Classifier | Test Accuracy | Log Loss | Best Hyperparameters |
|------------|:-------------:|:--------:|----------------------|
| **XGBoost** ✓ | **0.8532** | **0.3246** | n_estimators: 956, lr: 0.1266, max_depth: 8 |
| HistGradientBoosting | 0.8293 | 0.3529 | max_iter: 487, lr: 0.1391, max_leaf_nodes: 52 |
| CatBoost | 0.8251 | 0.3627 | iterations: 561, lr: 0.1792, depth: 8 |
| LightGBM | 0.8224 | 0.3680 | n_estimators: 326, lr: 0.0770, num_leaves: 58 |
| SGD | 0.7395 | 0.4928 | alpha: 0.0036 |

**XGBoost** achieves the highest accuracy (85.32%) and lowest log loss (0.3246). All gradient boosting models substantially outperform the linear SGD baseline, confirming the value of the engineered feature set.

---

## Saved Artifacts

The following files are written after training:

| File | Description |
|------|-------------|
| `winning_xgb_model.joblib` | Trained XGBoost classifier (compressed, `joblib`) |
| `scaler.joblib` | Fitted `StandardScaler` — required for inference |
| `tuning_results.json` | Accuracy, log loss, and best params for all five models |

---

## Inference

A standalone inference script (`inference.py`) reproduces the full preprocessing and feature engineering pipeline on CPU — no GPU required at prediction time.

**Load artifacts and predict:**

```python
from inference import load_artifacts, predict

model, scaler, w2v = load_artifacts()

result = predict(
    "How can I lose weight fast?",
    "What is the best diet to lose weight?",
    model, scaler, w2v
)
# {'is_duplicate': True, 'duplicate_probability': 0.7X, 'cleaned_q1': ..., 'cleaned_q2': ...}
```

**Sample predictions from notebook run:**

```
Q1: What is the step by step guide to invest in share market in india?
Q2: What is the step by step guide to invest in share market?
Duplicate: False  |  Probability: 0.4095

Q1: Do you believe there is life after death?
Q2: Is it true that there is life after death?
Duplicate: True   |  Probability: 0.5842
```

---

## Requirements

**Runtime:** Python 3.12 · CUDA 12 (GPU required for training; CPU sufficient for inference)

| Package | Version | Role |
|---------|---------|------|
| `numpy` | 2.0.2 | Numerical computing |
| `pandas` | 2.2.2 | Data manipulation |
| `matplotlib` | 3.10.0 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `tqdm` | 4.67.3 | Progress bars |
| `joblib` | 1.5.3 | Parallelism & model serialization |
| `scikit-learn` | 1.6.1 | HistGBM, StandardScaler, metrics |
| `cudf-cu12` | 26.2.1 | GPU DataFrames (RAPIDS) |
| `cupy-cuda12x` | 14.0.1 | GPU array backend (RAPIDS) |
| `xgboost` | 3.2.0 | Gradient boosting (GPU) |
| `lightgbm` | 4.6.0 | Gradient boosting (GPU) |
| `catboost` | 1.2.10 | Gradient boosting (GPU) |
| `optuna` | 4.8.0 | Bayesian hyperparameter tuning |
| `gensim` | 4.4.0 | Word2Vec embeddings |
| `nltk` | 3.9.1 | Stopword corpus |
| `rapidfuzz` | 3.14.5 | High-performance fuzzy matching |
| `fuzzywuzzy` | 0.18.0 | Fuzzy matching (legacy compatibility) |
| `Distance` | 0.1.3 | String distance utilities |
| `hvplot` | 0.12.2 | Interactive plots |
| `holoviews` | 1.22.1 | High-level plotting layer |
| `bokeh` | 3.8.2 | Plot rendering backend |

---

## Contributing

Contributions are welcome in the following areas:

- Improved preprocessing (stemming, lemmatization, subword tokenization)
- Additional feature engineering (TF-IDF overlap, BM25 similarity)
- Deep learning models (Siamese networks, sentence transformers)
- Inference API (FastAPI / Gradio wrapper)

Please open an issue before submitting a pull request to discuss the proposed change.

---

## Authors

| Name | GitHub |
|------|--------|
| Parit Kansal | [@ParitKansal](https://github.com/ParitKansal) |
| Pankhuri Kansal | [@Pankhuri9026](https://github.com/Pankhuri9026) |
