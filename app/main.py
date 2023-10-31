"""
FastAPI service for the Siamese LSTM duplicate-question detector.

Run:
    uvicorn app.main:app --reload --port 8000

Then POST to /predict:
    {"question1": "...", "question2": "..."}
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.lstm_inference import load_artifacts, predict

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, vocab, max_len, device = load_artifacts()
    _state["model"] = model
    _state["vocab"] = vocab
    _state["max_len"] = max_len
    _state["device"] = device
    print(f"Siamese LSTM loaded on {device} (vocab={len(vocab):,}, max_len={max_len})")
    yield
    _state.clear()


app = FastAPI(title="Quora Duplicate Question Detector (Siamese LSTM)", lifespan=lifespan)


class PredictRequest(BaseModel):
    question1: str = Field(..., min_length=1)
    question2: str = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    is_duplicate: bool
    duplicate_probability: float
    threshold_used: float
    cleaned_q1: str
    cleaned_q2: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    return predict(
        req.question1,
        req.question2,
        _state["model"],
        _state["vocab"],
        _state["max_len"],
        _state["device"],
        threshold=req.threshold,
    )
