"""
Inference script for Quora Duplicate Question Detection.
Runs the full preprocessing + feature engineering pipeline on two input sentences.

Saved artifacts required:
    winning_xgb_model.joblib  — trained XGBoost model
    scaler.joblib             — fitted StandardScaler
    word2vec-google-news-300  — loaded via gensim.downloader (cached after first run)

Usage:
    python inference.py
    or import predict() directly.
"""

import re
import string
import warnings

import joblib
import numpy as np
from rapidfuzz import fuzz
from nltk.corpus import stopwords
import nltk
import gensim.downloader as api

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not",
    "can't've": "can not have", "'cause": "because",
    "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "let's": "let us",
    "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "that'd": "that would",
    "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have",
    "there's": "there is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not",
    "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
    "we'll've": "we will have", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did",
    "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have", "you'd": "you would",
    "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are",
    "you've": "you have",
}

SYMBOL_REPLACEMENTS = {
    "%": " percent ",
    "$": " dollar ",
    "₹": " rupee ",
    "€": " euro ",
    "@": " at ",
    "[math]": "",
}

# All patterns compiled once at import
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
HTML_URL_PATTERN = re.compile(r"<.*?>|https?://\S+|www\.\S+")
PUNCTUATION_PATTERN = re.compile("[" + re.escape(string.punctuation) + "]")
WHITESPACE_PATTERN = re.compile(r"\s+")

STOPWORDS = set(stopwords.words("english"))

# Feature column order must exactly match training
_ORDERED_COLS = [
    "q1_char_len", "q2_char_len",
    "q1_token_no", "q2_token_no",
    "abs_len_diff", "mean_len",
    "min_char_len", "max_char_len",
    "total_words", "common_words", "ratio_of_common_to_total",
    "cwc_min", "cwc_max", "csc_min", "csc_max",
    "ctc_min", "ctc_max",
    "last_word_eq", "first_word_eq",
    "fuzz_ratio", "fuzz_partial_ratio",
    "token_sort_ratio", "token_set_ratio",
]


def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = EMOJI_PATTERN.sub("", text)
    text = HTML_URL_PATTERN.sub("", text)
    for old, new in SYMBOL_REPLACEMENTS.items():
        text = text.replace(old, new)
    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)
    text = PUNCTUATION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def _length_features(q1: str, q2: str, q1_toks: list, q2_toks: list) -> dict:
    q1_char = len(q1)
    q2_char = len(q2)
    q1_n = len(q1_toks)
    q2_n = len(q2_toks)
    return {
        "q1_char_len":  q1_char,
        "q2_char_len":  q2_char,
        "q1_token_no":  q1_n,
        "q2_token_no":  q2_n,
        "abs_len_diff": abs(q1_n - q2_n),
        "mean_len":     (q1_n + q2_n) / 2,
        "min_char_len": min(q1_char, q2_char),
        "max_char_len": max(q1_char, q2_char),
    }


def _token_features(q1_toks: list, q2_toks: list) -> dict:
    q1_set = set(q1_toks)
    q2_set = set(q2_toks)
    q1_sw = q1_set - STOPWORDS
    q2_sw = q2_set - STOPWORDS
    common     = q1_set & q2_set
    common_sw  = q1_sw & q2_sw
    total      = len(q1_set) + len(q2_set)
    n_common   = len(common)
    n_common_sw = len(common_sw)
    min_tok = min(len(q1_set), len(q2_set))
    max_tok = max(len(q1_set), len(q2_set))
    min_sw  = min(len(q1_sw), len(q2_sw))
    max_sw  = max(len(q1_sw), len(q2_sw))
    return {
        "total_words":              total,
        "common_words":             n_common,
        "ratio_of_common_to_total": n_common / (total + 1e-5),
        "cwc_min":   n_common_sw / (min_sw  + 1e-5),
        "cwc_max":   n_common_sw / (max_sw  + 1e-5),
        "csc_min":   n_common_sw / (min_tok + 1e-5),
        "csc_max":   n_common_sw / (max_tok + 1e-5),
        "ctc_min":   n_common    / (min_tok + 1e-5),
        "ctc_max":   n_common    / (max_tok + 1e-5),
        # index into the ordered lists, not a set, so first/last is deterministic
        "last_word_eq":  int(q1_toks[-1] == q2_toks[-1]) if q1_toks and q2_toks else 0,
        "first_word_eq": int(q1_toks[0]  == q2_toks[0])  if q1_toks and q2_toks else 0,
    }


def _fuzzy_features(q1: str, q2: str) -> dict:
    return {
        "fuzz_ratio":         fuzz.ratio(q1, q2),
        "fuzz_partial_ratio": fuzz.partial_ratio(q1, q2),
        "token_sort_ratio":   fuzz.token_sort_ratio(q1, q2),
        "token_set_ratio":    fuzz.token_set_ratio(q1, q2),
    }


def _build_feature_vector(q1: str, q2: str, w2v_model) -> np.ndarray:
    # Split once; pass token lists to all feature functions
    q1_toks = q1.split()
    q2_toks = q2.split()

    feats = {}
    feats.update(_length_features(q1, q2, q1_toks, q2_toks))
    feats.update(_token_features(q1_toks, q2_toks))
    feats.update(_fuzzy_features(q1, q2))

    scalar_vec = np.array([feats[c] for c in _ORDERED_COLS], dtype=np.float64)
    # pre_normalize=False matches training: plain np.mean with no L2 normalization
    q1_embed = w2v_model.get_mean_vector(q1_toks, ignore_missing=True, pre_normalize=False)
    q2_embed = w2v_model.get_mean_vector(q2_toks, ignore_missing=True, pre_normalize=False)
    return np.concatenate([scalar_vec, q1_embed, q2_embed])


def _load_joblib(path: str, save_hint: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File '{path}' not found.\n"
            f"Save it from your notebook with:\n  {save_hint}"
        )


def load_artifacts(
    model_path: str = "winning_xgb_model.joblib",
    scaler_path: str = "scaler.joblib",
    w2v_model_name: str = "word2vec-google-news-300",
):
    print("Loading model...")
    model = _load_joblib(model_path, f"joblib.dump(best_clf, '{model_path}', compress=3)")
    print("Loading scaler...")
    scaler = _load_joblib(scaler_path, f"joblib.dump(scaler, '{scaler_path}', compress=3)")
    print("Loading Word2Vec (uses local cache after first download)...")
    w2v = api.load(w2v_model_name)
    print("All artifacts loaded.")
    return model, scaler, w2v


def predict(
    sentence1: str,
    sentence2: str,
    model,
    scaler,
    w2v_model,
) -> dict:
    q1 = preprocess(sentence1)
    q2 = preprocess(sentence2)
    x_scaled = scaler.transform(_build_feature_vector(q1, q2, w2v_model).reshape(1, -1))
    proba = model.predict_proba(x_scaled)[0]
    return {
        "is_duplicate":          bool(np.argmax(proba)),
        "duplicate_probability": round(float(proba[1]), 4),
        "cleaned_q1":            q1,
        "cleaned_q2":            q2,
    }


if __name__ == "__main__":
    model, scaler, w2v = load_artifacts()

    pairs = [
        (
            "What is the step by step guide to invest in share market in india?",
            "What is the step by step guide to invest in share market?",
        ),
        (
            "Do you believe there is life after death?",
            "Is it true that there is life after death?",
        ),
    ]

    print("\n" + "=" * 70)
    for q1, q2 in pairs:
        result = predict(q1, q2, model, scaler, w2v)
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print(f"Duplicate: {result['is_duplicate']}  |  Probability: {result['duplicate_probability']:.4f}")
        print("-" * 70)
