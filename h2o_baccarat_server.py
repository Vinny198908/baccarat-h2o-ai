#!/usr/bin/env python3
"""H2O AutoML training/scoring server for the All-In Baccarat app.

Run locally or on an HTTPS server. The browser app calls POST /train and POST /score.
This software estimates patterns from historical data; it cannot guarantee future baccarat outcomes.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import h2o
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from h2o.automl import H2OAutoML

APP_DIR = Path(os.environ.get("BACCARAT_MODEL_DIR", "./baccarat_h2o_models")).resolve()
APP_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = os.environ.get("BACCARAT_API_KEY", "").strip()
MAX_MODELS = int(os.environ.get("H2O_MAX_MODELS", "20"))
MAX_RUNTIME_SECS = int(os.environ.get("H2O_MAX_RUNTIME_SECS", "180"))
MIN_PREFIX = int(os.environ.get("BACCARAT_MIN_PREFIX", "6"))

app = FastAPI(title="Baccarat H2O Predictor", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("BACCARAT_CORS_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_lock = threading.Lock()
_model = None
_model_id = None
_model_classes: list[str] = []
_meta: dict[str, Any] = {}

FEATURE_COLUMNS = [
    "hand_count", "player_count", "banker_count", "tie_count",
    "recent_player", "recent_banker", "recent_tie", "streak_length",
    "streak_side", "alternations_20", "player_rate", "banker_rate", "tie_rate",
    *[f"lag_{i}" for i in range(1, 13)],
]
CATEGORICAL = ["streak_side", *[f"lag_{i}" for i in range(1, 13)]]


def ensure_h2o() -> None:
    try:
        h2o.connection()
    except Exception:
        h2o.init(nthreads=-1, min_mem_size="1G", max_mem_size=os.environ.get("H2O_MAX_MEM", "4G"))


def auth(request: Request) -> None:
    if not API_KEY:
        return
    got = request.headers.get("authorization", "")
    if got != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def clean_sequence(seq: str) -> str:
    seq = "".join(c for c in (seq or "").upper() if c in "PBT")
    return seq


def features_for_sequence(seq: str) -> dict[str, Any]:
    seq = clean_sequence(seq)
    recent = seq[-20:]
    last_non_tie = next((c for c in reversed(seq) if c in "PB"), "N")
    streak = 0
    if last_non_tie != "N":
        for c in reversed(seq):
            if c == "T":
                continue
            if c == last_non_tie:
                streak += 1
            else:
                break
    non_tie_recent = [c for c in recent if c in "PB"]
    alternations = sum(non_tie_recent[i] != non_tie_recent[i-1] for i in range(1, len(non_tie_recent)))
    n = len(seq) or 1
    row: dict[str, Any] = {
        "hand_count": len(seq),
        "player_count": seq.count("P"),
        "banker_count": seq.count("B"),
        "tie_count": seq.count("T"),
        "recent_player": recent.count("P"),
        "recent_banker": recent.count("B"),
        "recent_tie": recent.count("T"),
        "streak_length": streak,
        "streak_side": last_non_tie,
        "alternations_20": alternations,
        "player_rate": seq.count("P") / n,
        "banker_rate": seq.count("B") / n,
        "tie_rate": seq.count("T") / n,
    }
    for i in range(1, 13):
        row[f"lag_{i}"] = seq[-i] if len(seq) >= i else "N"
    return row


def rows_from_shoes(shoes: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for shoe_idx, shoe in enumerate(shoes):
        seq = clean_sequence(str(shoe.get("sequence", "")))
        if len(seq) <= MIN_PREFIX:
            continue
        sid = str(shoe.get("id") or f"shoe_{shoe_idx}")
        for cut in range(MIN_PREFIX, len(seq)):
            row = features_for_sequence(seq[:cut])
            row["target"] = seq[cut]
            row["shoe_id"] = sid
            rows.append(row)
    return pd.DataFrame(rows)


def load_latest() -> None:
    global _model, _model_id, _model_classes, _meta
    ensure_h2o()
    meta_path = APP_DIR / "latest.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
        model_path = meta.get("model_path")
        if model_path and Path(model_path).exists():
            _model = h2o.load_model(model_path)
            _model_id = _model.model_id
            _model_classes = list(meta.get("classes", []))
            _meta = meta
    except Exception as exc:
        print("Could not load latest model:", exc)


def prediction_probabilities(model, frame, classes: list[str]) -> dict[str, float]:
    pred = model.predict(frame).as_data_frame(use_multi_thread=True)
    row = pred.iloc[0]
    # H2O classification prediction columns are predict plus one probability column per response level.
    prob_cols = [c for c in pred.columns if c != "predict"]
    probs: dict[str, float] = {"P": 0.0, "B": 0.0, "T": 0.0}
    if len(prob_cols) == len(classes):
        for cls, col in zip(classes, prob_cols):
            probs[cls] = float(row[col])
    else:
        # Fallback for builds that name probability columns by class labels.
        for cls in classes:
            if cls in pred.columns:
                probs[cls] = float(row[cls])
    total = sum(probs.values())
    if total <= 0:
        chosen = str(row.get("predict", "P"))
        probs[chosen] = 1.0
        total = 1.0
    return {k: v / total for k, v in probs.items()}


@app.on_event("startup")
def startup() -> None:
    load_latest()



@app.get("/")
def app_home():
    return FileResponse(Path(__file__).with_name("index.html"))

@app.get("/training")
def training_home():
    return FileResponse(Path(__file__).with_name("training.html"))

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_id": _model_id, "classes": _model_classes, "meta": _meta}


@app.post("/train")
def train(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    shoes = payload.get("shoes") or []
    if len(shoes) < 5:
        raise HTTPException(400, "At least 5 saved shoes are required")
    df = rows_from_shoes(shoes)
    if len(df) < 40:
        raise HTTPException(400, "Not enough training hands after feature creation")
    unique_shoes = list(dict.fromkeys(df["shoe_id"].tolist()))
    if len(unique_shoes) < 5:
        raise HTTPException(400, "Need at least 5 usable unique shoes")
    split = max(1, int(round(len(unique_shoes) * 0.2)))
    valid_ids = set(unique_shoes[-split:])
    train_df = df[~df.shoe_id.isin(valid_ids)].copy()
    valid_df = df[df.shoe_id.isin(valid_ids)].copy()
    classes = sorted(train_df["target"].unique().tolist())
    if len(classes) < 2:
        raise HTTPException(400, "Training data needs at least two outcome classes")

    ensure_h2o()
    with _lock:
        hf_train = h2o.H2OFrame(train_df[FEATURE_COLUMNS + ["target"]])
        hf_valid = h2o.H2OFrame(valid_df[FEATURE_COLUMNS + ["target"]])
        for col in CATEGORICAL + ["target"]:
            hf_train[col] = hf_train[col].asfactor()
            hf_valid[col] = hf_valid[col].asfactor()
        aml = H2OAutoML(
            max_models=MAX_MODELS,
            max_runtime_secs=MAX_RUNTIME_SECS,
            seed=20260809,
            sort_metric="logloss",
            balance_classes=True,
            exclude_algos=["DeepLearning"],
        )
        aml.train(x=FEATURE_COLUMNS, y="target", training_frame=hf_train, leaderboard_frame=hf_valid)
        leader = aml.leader
        model_path = h2o.save_model(leader, path=str(APP_DIR), force=True)

        # Held-out whole-shoe accuracy.
        pred = leader.predict(hf_valid).as_data_frame(use_multi_thread=True)
        truth = valid_df["target"].reset_index(drop=True).astype(str)
        accuracy = float((pred["predict"].astype(str).reset_index(drop=True) == truth).mean())
        domain = list(leader._model_json.get("output", {}).get("domains", [])[-1] or classes)

        global _model, _model_id, _model_classes, _meta
        _model, _model_id, _model_classes = leader, leader.model_id, domain
        _meta = {
            "model_id": leader.model_id,
            "model_path": model_path,
            "classes": domain,
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "training_shoes": int(len(unique_shoes) - len(valid_ids)),
            "validation_shoes": int(len(valid_ids)),
            "accuracy": accuracy,
            "leaderboard": aml.leaderboard.head(rows=5).as_data_frame().to_dict(orient="records"),
        }
        (APP_DIR / "latest.json").write_text(json.dumps(_meta, indent=2, default=str))
        return _meta


@app.post("/score")
def score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    if _model is None:
        raise HTTPException(409, "No trained model. POST your library to /train first.")
    features = payload.get("features") or payload
    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    if len(seq) < MIN_PREFIX:
        raise HTTPException(400, f"Need at least {MIN_PREFIX} entered hands before scoring")
    row = features_for_sequence(seq)
    ensure_h2o()
    hf = h2o.H2OFrame(pd.DataFrame([row], columns=FEATURE_COLUMNS))
    for col in CATEGORICAL:
        hf[col] = hf[col].asfactor()
    probs = prediction_probabilities(_model, hf, _model_classes)
    return {
        "p": probs["P"], "b": probs["B"], "t": probs["T"],
        "model_id": _model_id,
        "classes": _model_classes,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
