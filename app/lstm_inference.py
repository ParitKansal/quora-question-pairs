"""
Inference module for the Siamese LSTM duplicate-question detector.

Saved artifacts required (see notebook section 21 — Winning Model Serialization):
    artifacts/siamese_model.pt     — trained SiameseEncoder state_dict
    artifacts/siamese_vocab.json   — vocabulary + max_len/hidden_dim config
"""

import json
import re
import string
from pathlib import Path

import torch
import torch.nn as nn

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# ---------------------------------------------------------------------------
# Text cleaning — must exactly mirror the notebook's preprocess_gpu (section 8),
# since that's what question1/question2 were cleaned with before training.
# ---------------------------------------------------------------------------
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

PUNCTUATION_PATTERN = re.compile("[" + re.escape(string.punctuation) + "]")


def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    for old, new in SYMBOL_REPLACEMENTS.items():
        text = text.replace(old, new)
    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)
    text = PUNCTUATION_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Model architecture — must exactly match SiameseEncoder from the notebook's
# training cell (section 20), since we're loading its state_dict into this.
# ---------------------------------------------------------------------------
class SiameseEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding_matrix, dtype=torch.float32), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        combined_dim = hidden_dim * 2 * 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        return torch.cat([h_n[0], h_n[1]], dim=1)

    def forward(self, q1, q2):
        h1 = self.encode(q1)
        h2 = self.encode(q2)
        combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1)
        return self.classifier(combined).squeeze(1)


def encode_tokens(text: str, vocab: dict, max_len: int) -> list:
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]
    ids = [vocab.get(tok, unk_idx) for tok in text.split()[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


def load_artifacts(
    model_path: Path = ARTIFACTS_DIR / "siamese_model.pt",
    vocab_path: Path = ARTIFACTS_DIR / "siamese_vocab.json",
    embed_dim: int = 300,
):
    try:
        with open(vocab_path) as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{vocab_path}' not found. This file is only written by the notebook's "
            "Winning Model Serialization step if the Siamese LSTM was the true best model."
        )
    vocab = meta["vocab"]
    max_len = meta["max_len"]
    hidden_dim = meta["hidden_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{model_path}' not found alongside '{vocab_path}'.")

    # The embedding matrix values don't matter here — load_state_dict overwrites them;
    # we just need a correctly-shaped placeholder to construct the module.
    placeholder_embeddings = torch.zeros(len(vocab), embed_dim)
    model = SiameseEncoder(len(vocab), embed_dim, hidden_dim, placeholder_embeddings)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, vocab, max_len, device


def predict(
    sentence1: str,
    sentence2: str,
    model,
    vocab: dict,
    max_len: int,
    device,
    threshold: float = 0.5,
) -> dict:
    q1 = preprocess(sentence1)
    q2 = preprocess(sentence2)

    q1_ids = torch.tensor([encode_tokens(q1, vocab, max_len)], dtype=torch.long).to(device)
    q2_ids = torch.tensor([encode_tokens(q2, vocab, max_len)], dtype=torch.long).to(device)

    with torch.no_grad():
        logit = model(q1_ids, q2_ids)
        prob = torch.sigmoid(logit).item()

    return {
        "is_duplicate": prob >= threshold,
        "duplicate_probability": prob,
        "threshold_used": threshold,
        "cleaned_q1": q1,
        "cleaned_q2": q2,
    }
