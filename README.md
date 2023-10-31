# Quora Question Pairs — Duplicate Detection

Detects whether two Quora questions are duplicates. A GPU-accelerated training notebook compares five tuned statistical models against a Siamese bidirectional LSTM; the LSTM won on this run's held-out test set (**F1 0.7986**), and its trained weights ship in this repo behind a ready-to-run FastAPI service.

- **Training notebook:** [`Quora_Question_Pairs.ipynb`](Quora_Question_Pairs.ipynb) (24 sections, end-to-end)
- **Inference service:** [`app/`](app/) — FastAPI wrapper around the trained Siamese LSTM, CPU-only, no retraining needed

---

## Table of Contents

1. [Quickstart — Run the API](#quickstart--run-the-api)
2. [Results](#results)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Training Pipeline](#training-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Models & Selection](#models--selection)
8. [Saved Artifacts & Inference](#saved-artifacts--inference)
9. [Reproducing Training](#reproducing-training)
10. [API Reference](#api-reference)
11. [Contributing](#contributing)
12. [Authors](#authors)

---

## Quickstart — Run the API

The trained model (`artifacts/siamese_model.pt` + `artifacts/siamese_vocab.json`) is committed to this repo, so you can serve predictions without running the notebook. Python 3.12 is recommended (PyTorch wheels lag very new Python releases).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r app/requirements.txt      # fastapi, uvicorn, torch — nothing else
.venv/bin/uvicorn app.main:app --port 8000
```

Test it:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question1": "Do you believe there is life after death?", "question2": "Is it true that there is life after death?"}'
```

```json
{
  "is_duplicate": true,
  "duplicate_probability": 1.0,
  "threshold_used": 0.5,
  "cleaned_q1": "do you believe there is life after death",
  "cleaned_q2": "is it true that there is life after death"
}
```

Interactive Swagger docs at `http://127.0.0.1:8000/docs`. Full endpoint details in [API Reference](#api-reference).

---

## Results

All six candidates evaluated on the same held-out test split (~61k rows):

| Model | F1 | Precision | Recall | AUC | Accuracy | LogLoss |
|-------|:--:|:---------:|:------:|:---:|:--------:|:-------:|
| **SiameseLSTM** 🏆 | **0.7986** | 0.7919 | **0.8053** | 0.9181 | **0.8508** | 1.0253 |
| LightGBM | 0.7776 | 0.7802 | 0.7750 | **0.9201** | 0.8372 | **0.3404** |
| HistGBM | 0.7736 | 0.7774 | 0.7698 | 0.9183 | 0.8345 | 0.3444 |
| XGBoost | 0.7735 | 0.7736 | 0.7734 | 0.9175 | 0.8337 | 0.3508 |
| CatBoost | 0.7413 | 0.7387 | 0.7439 | 0.8945 | 0.8093 | 0.3911 |
| SGD | 0.6184 | 0.6643 | 0.5784 | 0.8243 | 0.7378 | 0.4920 |

The Siamese LSTM wins on F1, Precision, Recall, and Accuracy — nearly 2 F1 points over the best statistical model. LightGBM keeps the best AUC and Log Loss, meaning its probability *rankings* are better calibrated even though its default-threshold classification is weaker.

SiameseLSTM confusion matrix (rows = actual, columns = predicted):

```
              Pred 0   Pred 1
Actual 0      33659     4713
Actual 1       4336    17936
```

Exact metrics vary run-to-run (Optuna trials, seeds, epochs). Each notebook run writes a full per-model report to `tuning_results.json`.

---

## Project Structure

```
quora-question-pairs/
├── Quora_Question_Pairs.ipynb   # end-to-end training notebook (24 sections)
├── README.md
├── requirements-training.txt    # full training environment (Colab/RAPIDS, GPU)
├── .gitignore
│
├── artifacts/                   # trained models, committed — ready for inference
│   ├── siamese_model.pt         #   Siamese LSTM weights (this run's winner)
│   ├── siamese_vocab.json       #   LSTM vocabulary + max_len/hidden_dim config
│   ├── winning_xgb_model.joblib #   best statistical model (filename is legacy; may hold any of the 5 candidates)
│   └── scaler.joblib            #   fitted StandardScaler for the 623-dim tabular features
│
├── app/                         # standalone FastAPI inference service (Siamese LSTM)
│   ├── main.py                  #   FastAPI app: /health, /predict
│   ├── lstm_inference.py        #   preprocessing + model loading + predict()
│   ├── requirements.txt         #   minimal deps: fastapi, uvicorn, torch
│   └── __init__.py
│
└── data/                        # local dataset cache (gitignored, populated by kagglehub)
```

**Why two requirements files?** `requirements-training.txt` is the heavy GPU environment needed to *run the notebook* (RAPIDS, XGBoost/LightGBM/CatBoost, Optuna, gensim, PyTorch). `app/requirements.txt` is the minimal set needed to *serve* the trained LSTM (`fastapi`, `uvicorn`, `torch` — nothing else).

---

## Dataset

**Source:** [Kaggle — Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs)

404,290 rows (404,287 after dropping 3 nulls), downloaded automatically by the notebook via `kagglehub`.

| Column | Description |
|--------|-------------|
| `id` | Unique pair identifier |
| `qid1`, `qid2` | Individual question identifiers |
| `question1`, `question2` | Raw question text |
| `is_duplicate` | Label — `1` if duplicate, `0` otherwise |

---

## Training Pipeline

```
Raw CSV
  └── EDA (shape, nulls, class distribution)
        └── GPU Preprocessing (cudf, with CPU pandas fallback)
              └── Feature Engineering
                    ├── Length features       (cudf, GPU)
                    ├── Token features        (joblib, CPU parallel)
                    ├── Fuzzy features        (rapidfuzz, CPU parallel)
                    └── Word2Vec embeddings   (multiprocessing, CPU parallel)
                          └── Train / Valid / Test Split (70 / 15 / 15)
                                └── StandardScaler
                                      ├── Optuna Tuning (5 statistical models, F1-driven)
                                      │       └── Threshold Selection (Youden's J)
                                      └── Siamese LSTM Training (PyTorch, raw text, same split)
                                            └── Cross-architecture F1 comparison → Save Artifacts
```

Every stage is optimized for the 404k-row dataset:

| Stage | Technique |
|-------|-----------|
| Text preprocessing | RAPIDS `cudf` string ops on GPU (transparent CPU pandas fallback) |
| Token features | `joblib.Parallel(prefer='threads')` across all CPU cores (~9,000 rows/sec) |
| Fuzzy matching | `rapidfuzz` (GIL-free) + chunked threading via `joblib` |
| Word2Vec embedding | `multiprocessing.Pool`, one process per core (~37 sec per question column) |
| Hyperparameter tuning | Optuna TPE + `MedianPruner`, `eval_set` early stopping |
| Model training | XGBoost/LightGBM/CatBoost on GPU; PyTorch LSTM on CUDA |

**Preprocessing** (identical at training and inference time — mirrored in [`app/lstm_inference.py`](app/lstm_inference.py)): lowercase and strip, remove emojis, strip HTML/URLs, expand symbols (`%` → `percent`, `$` → `dollar`, …), expand 100+ English contractions, remove punctuation, collapse whitespace.

---

## Feature Engineering

**For the statistical models only** — the final tabular matrix is **623 columns**. The Siamese LSTM skips all of this and consumes the cleaned raw text directly.

| Group | Columns | Compute | Features |
|-------|:-------:|---------|----------|
| Length | 8 | GPU (`cudf`) | char/token lengths per question, `abs_len_diff`, `mean_len`, min/max char length |
| Token | 11 | CPU parallel (`joblib`) | `common_words`, `total_words`, common/total ratio, `cwc`/`csc`/`ctc` min-max ratios, first/last word match |
| Fuzzy | 4 | CPU parallel (`rapidfuzz`) | `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `token_set_ratio` |
| Word2Vec | 600 | CPU parallel (`multiprocessing`) | mean of `word2vec-google-news-300` token vectors per question (300 dims × 2 questions) |

**Siamese LSTM input:** cleaned text → whitespace tokenization → vocabulary of tokens appearing ≥ 4 times (`MIN_FREQ`) → pad/truncate to 40 tokens (`MAX_LEN`). Token embeddings are initialized from the same Word2Vec vectors, then fine-tuned during training (`freeze=False`).

---

## Models & Selection

**Split:** 70% train / 15% validation / 15% test (~282k / ~61k / ~61k rows). The same row split is reused for both model families, so results are directly comparable.

**Statistical models** — LightGBM, XGBoost, CatBoost, HistGradientBoosting, SGD — each tuned with **Optuna** (TPE sampler + `MedianPruner`) optimizing validation F1, with `eval_set` early stopping for XGBoost/LightGBM, then refit and evaluated on the test split. Since the default 0.5 probability cutoff is arbitrary, a better decision threshold is selected for the best statistical model by maximizing Youden's J (`TPR − FPR`) on the validation ROC curve (section 19).

**Siamese LSTM** — a bidirectional LSTM encoder shared across both questions (`SiameseEncoder`) produces two sentence encodings, combined as `[h1, h2, |h1−h2|, h1×h2]` and passed through a small MLP. Trained with `BCEWithLogitsLoss` + Adam; the checkpoint with the best validation F1 is kept (section 20).

**Winner selection** (section 21) — the true winner is `max(results, key=F1)` across all 6 candidates:

- The best **statistical** model is always saved (`winning_xgb_model.joblib` + `scaler.joblib`) for the tabular-feature inference path.
- **If the LSTM is the overall winner** — as in this run — its weights and vocabulary are additionally saved (`siamese_model.pt` + `siamese_vocab.json`) for its own raw-text inference path.
- `tuning_results.json` records every candidate's metrics, best hyperparameters, the tuned threshold, and the declared winner.

---

## Saved Artifacts & Inference

| File | Description | In repo? |
|------|-------------|:---:|
| `artifacts/siamese_model.pt` | Trained `SiameseEncoder` state dict (this run's winner) | ✅ |
| `artifacts/siamese_vocab.json` | LSTM vocabulary + `max_len`/`hidden_dim` config | ✅ |
| `artifacts/winning_xgb_model.joblib` | Best statistical classifier — despite the legacy filename, may be any of the five candidates | ✅ |
| `artifacts/scaler.joblib` | Fitted `StandardScaler` for the 623-dim tabular features | ✅ |
| `tuning_results.json` | Per-model metrics + hyperparameters + winner | regenerated each notebook run, not committed |

There are **two separate, non-interchangeable inference paths**, since the two model families consume different inputs:

1. **Siamese LSTM** — [`app/lstm_inference.py`](app/lstm_inference.py): text cleaning + tokenization + PyTorch model, CPU or GPU. This is what the API serves, and it works directly from Python too:

   ```python
   from app.lstm_inference import load_artifacts, predict

   model, vocab, max_len, device = load_artifacts()
   result = predict(
       "Do you believe there is life after death?",
       "Is it true that there is life after death?",
       model, vocab, max_len, device,
   )
   # {'is_duplicate': True, 'duplicate_probability': 1.0, 'threshold_used': 0.5, ...}
   ```

2. **Statistical model** — reproduces the full 623-dim feature pipeline on CPU. The code for a standalone `inference.py` lives in **notebook section 23**; it isn't shipped as a repo file since the LSTM won this run.

---

## Reproducing Training

Training needs a CUDA GPU (Colab or similar) for RAPIDS, GPU gradient boosting, and PyTorch:

```bash
pip install -r requirements-training.txt
```

Then run [`Quora_Question_Pairs.ipynb`](Quora_Question_Pairs.ipynb) top to bottom. Notes:

- **Kaggle credentials** — section 2 sets up the Kaggle API token; the dataset then downloads automatically via `kagglehub` (section 4).
- **Word2Vec** — `word2vec-google-news-300` (1.6 GB) downloads automatically via `gensim.downloader` on first run and is cached locally.
- **RAPIDS is optional** — if `cudf`/`cupy` are unavailable, preprocessing transparently falls back to CPU pandas.
- Key dependency groups in [`requirements-training.txt`](requirements-training.txt): gradient boosting (`xgboost`, `lightgbm`, `catboost`), tuning (`optuna`), NLP (`gensim`, `nltk`, `rapidfuzz`), deep learning (`torch`), and optionally RAPIDS (`cudf-cu12`, `cupy-cuda12x`). See the file for exact pins.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness check; confirms the model is loaded |
| `/predict` | POST | Classify a question pair |

**`POST /predict` request body:**

```json
{"question1": "string", "question2": "string", "threshold": 0.5}
```

`threshold` is optional (default `0.5`) — the notebook's Youden's-J-tuned threshold is calibrated for the statistical model, not the LSTM, so it isn't reused here.

**Response:** `is_duplicate` (bool), `duplicate_probability` (float), `threshold_used` (float), plus the cleaned versions of both questions.

Sample predictions from the shipped model:

```
Q1: Do you believe there is life after death?
Q2: Is it true that there is life after death?
  -> duplicate=True   prob=1.0000

Q1: How do I learn Python?
Q2: What is the best way to bake a chocolate cake?
  -> duplicate=False  prob=0.0000

Q1: What is the step by step guide to invest in share market in india?
Q2: What is the step by step guide to invest in share market?
  -> duplicate=False  prob=0.0133   # a genuinely hard near-duplicate case
```

---

## Contributing

Welcome areas: a proper threshold-selection pass for the LSTM (currently fixed at 0.5), containerizing `app/` (Dockerfile), a batch-prediction endpoint, sentence-transformer / cross-encoder baselines, and richer features (TF-IDF overlap, BM25 similarity). Please open an issue before submitting a pull request.

---

## Authors

| Name | GitHub |
|------|--------|
| Parit Kansal | [@ParitKansal](https://github.com/ParitKansal) |
| Pankhuri Kansal | [@Pankhuri9026](https://github.com/Pankhuri9026) |
