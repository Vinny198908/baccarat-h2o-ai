#!/usr/bin/env python3
"""H2O AutoML baccarat scoring server with roadmap/pattern features.

The browser app POSTs saved shoes to /train and the current sequence to /score.
The model uses only information available before the next hand; it cannot know future cards.
"""
from __future__ import annotations

import json
import os
import threading
import re
import gzip
import time
from collections import OrderedDict
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import h2o
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from h2o.automl import H2OAutoML

APP_DIR = Path(os.environ.get("BACCARAT_MODEL_DIR", "./baccarat_h2o_models")).resolve()
APP_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = os.environ.get("BACCARAT_API_KEY", "").strip()
MAX_MODELS = int(os.environ.get("H2O_MAX_MODELS", "30"))
MAX_RUNTIME_SECS = int(os.environ.get("H2O_MAX_RUNTIME_SECS", "240"))
MIN_PREFIX = int(os.environ.get("BACCARAT_MIN_PREFIX", "6"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip() or "deepseek-v4-pro"
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions").strip()
DEEPSEEK_TIMEOUT_SECS = int(os.environ.get("DEEPSEEK_TIMEOUT_SECS", "45"))
COMPACT_LIBRARY_PATH = Path(os.environ.get("BACCARAT_COMPACT_LIBRARY", "./baccarat_shoes_compact.txt.gz")).resolve()
LIBRARY_MIN_SUFFIX = int(os.environ.get("BACCARAT_LIBRARY_MIN_SUFFIX", "4"))
LIBRARY_MAX_SUFFIX = int(os.environ.get("BACCARAT_LIBRARY_MAX_SUFFIX", "12"))
LIBRARY_MIN_MATCHES = int(os.environ.get("BACCARAT_LIBRARY_MIN_MATCHES", "30"))
LIBRARY_CACHE_SIZE = int(os.environ.get("BACCARAT_LIBRARY_CACHE_SIZE", "128"))

app = FastAPI(title="Baccarat H2O + DeepSeek Road Intelligence", version="4.1")
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
_library_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_library_cache_lock = threading.Lock()

BASE_NUMERIC = [
    "hand_count", "player_count", "banker_count", "tie_count",
    "recent_player", "recent_banker", "recent_tie", "streak_length",
    "alternations_20", "player_rate", "banker_rate", "tie_rate",
    "big_columns", "big_last_col_depth", "big_prev_col_depth", "big_max_depth",
    "big_mean_depth", "big_single_cols", "big_deep_cols", "big_last4_depth_sum",
    "non_tie_runs", "avg_run_length", "max_run_length", "recent_run_mean",
]
DERIVED_NUMERIC = []
for prefix in ("eye", "small", "roach"):
    DERIVED_NUMERIC += [
        f"{prefix}_count", f"{prefix}_red_rate", f"{prefix}_recent_red_rate",
        f"{prefix}_streak_length", f"{prefix}_alternations_12",
    ]
PREVIEW_NUMERIC = [
    "preview_p_red_count", "preview_b_red_count", "preview_difference",
    "preview_same_count", "preview_opposite_count",
]
FEATURE_COLUMNS = (
    BASE_NUMERIC
    + DERIVED_NUMERIC
    + PREVIEW_NUMERIC
    + ["streak_side", "pattern_type", "eye_last", "small_last", "roach_last"]
    + [f"preview_p_{x}" for x in ("eye", "small", "roach")]
    + [f"preview_b_{x}" for x in ("eye", "small", "roach")]
    + [f"lag_{i}" for i in range(1, 21)]
)
CATEGORICAL = [
    "streak_side", "pattern_type", "eye_last", "small_last", "roach_last",
    *[f"preview_p_{x}" for x in ("eye", "small", "roach")],
    *[f"preview_b_{x}" for x in ("eye", "small", "roach")],
    *[f"lag_{i}" for i in range(1, 21)],
]


def ensure_h2o() -> None:
    try:
        h2o.connection()
    except Exception:
        # Keep H2O optional and memory-bounded for small Render instances.
        # The full 200k-shoe library is streamed from gzip and is NOT loaded into H2O RAM.
        h2o.init(nthreads=int(os.environ.get("H2O_NTHREADS", "2")), max_mem_size=os.environ.get("H2O_MAX_MEM", "384M"))


def auth(request: Request) -> None:
    if not API_KEY:
        return
    got = request.headers.get("authorization", "")
    if got != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in "PBT")


def big_road_columns(seq: str) -> list[list[str]]:
    cols: list[list[str]] = []
    for c in clean_sequence(seq):
        if c == "T":
            continue
        if not cols or cols[-1][0] != c:
            cols.append([c])
        else:
            cols[-1].append(c)
    return cols


def derive_road(seq: str, offset: int) -> list[str]:
    cols = big_road_columns(seq)
    marks: list[str] = []
    if len(cols) <= offset:
        return marks
    for c in range(offset, len(cols)):
        current_len = len(cols[c])
        start_row = 1 if c == offset else 0
        for r in range(start_row, current_len):
            if r == 0:
                near_len = len(cols[c - 1]) if c - 1 >= 0 else 0
                far_idx = c - 1 - offset
                far_len = len(cols[far_idx]) if far_idx >= 0 else 0
                red = near_len == far_len
            else:
                ref_idx = c - offset
                ref_len = len(cols[ref_idx]) if ref_idx >= 0 else 0
                same_row_exists = ref_len >= r + 1
                above_row_exists = ref_len >= r
                red = same_row_exists == above_row_exists
            marks.append("R" if red else "B")
    return marks


def run_lengths(seq: str) -> list[int]:
    s = [c for c in clean_sequence(seq) if c in "PB"]
    if not s:
        return []
    out, n = [], 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            n += 1
        else:
            out.append(n)
            n = 1
    out.append(n)
    return out


def derived_stats(marks: list[str], prefix: str) -> dict[str, Any]:
    if not marks:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_red_rate": 0.5,
            f"{prefix}_recent_red_rate": 0.5,
            f"{prefix}_streak_length": 0,
            f"{prefix}_alternations_12": 0,
            f"{prefix}_last": "N",
        }
    recent = marks[-12:]
    streak = 1
    for i in range(len(marks)-2, -1, -1):
        if marks[i] == marks[-1]:
            streak += 1
        else:
            break
    alt = sum(recent[i] != recent[i-1] for i in range(1, len(recent)))
    return {
        f"{prefix}_count": len(marks),
        f"{prefix}_red_rate": marks.count("R") / len(marks),
        f"{prefix}_recent_red_rate": recent.count("R") / len(recent),
        f"{prefix}_streak_length": streak,
        f"{prefix}_alternations_12": alt,
        f"{prefix}_last": marks[-1],
    }


def preview_mark(seq: str, result: str, offset: int) -> str:
    before = derive_road(seq, offset)
    after = derive_road(seq + result, offset)
    return after[-1] if len(after) > len(before) else "N"


def pattern_type(non_tie_recent: list[str], streak: int, alternations: int) -> str:
    if streak >= 4:
        return "STREAK"
    if len(non_tie_recent) >= 6 and alternations >= max(4, len(non_tie_recent)-3):
        return "CHOP"
    if len(non_tie_recent) >= 8:
        runs = run_lengths("".join(non_tie_recent))[-6:]
        if runs and max(runs) <= 2:
            return "TIGHT_CHOP"
        if runs and sum(r >= 3 for r in runs) >= 2:
            return "RUNS"
    return "MIXED"


def features_for_sequence(seq: str) -> dict[str, Any]:
    seq = clean_sequence(seq)
    recent = seq[-20:]
    non_tie_recent = [c for c in recent if c in "PB"]
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
    alternations = sum(non_tie_recent[i] != non_tie_recent[i-1] for i in range(1, len(non_tie_recent)))
    n = len(seq) or 1
    cols = big_road_columns(seq)
    depths = [len(c) for c in cols]
    runs = run_lengths(seq)
    recent_runs = runs[-6:]
    eye, small, roach = derive_road(seq, 1), derive_road(seq, 2), derive_road(seq, 3)

    pprev = {name: preview_mark(seq, "P", off) for name, off in (("eye",1),("small",2),("roach",3))}
    bprev = {name: preview_mark(seq, "B", off) for name, off in (("eye",1),("small",2),("roach",3))}
    p_red = sum(v == "R" for v in pprev.values())
    b_red = sum(v == "R" for v in bprev.values())
    same = sum(pprev[k] == bprev[k] and pprev[k] != "N" for k in pprev)
    opposite = sum(pprev[k] != bprev[k] and pprev[k] != "N" and bprev[k] != "N" for k in pprev)

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
        "big_columns": len(cols),
        "big_last_col_depth": depths[-1] if depths else 0,
        "big_prev_col_depth": depths[-2] if len(depths) >= 2 else 0,
        "big_max_depth": max(depths) if depths else 0,
        "big_mean_depth": (sum(depths) / len(depths)) if depths else 0.0,
        "big_single_cols": sum(d == 1 for d in depths),
        "big_deep_cols": sum(d >= 4 for d in depths),
        "big_last4_depth_sum": sum(depths[-4:]),
        "non_tie_runs": len(runs),
        "avg_run_length": (sum(runs)/len(runs)) if runs else 0.0,
        "max_run_length": max(runs) if runs else 0,
        "recent_run_mean": (sum(recent_runs)/len(recent_runs)) if recent_runs else 0.0,
        "pattern_type": pattern_type(non_tie_recent, streak, alternations),
        "preview_p_red_count": p_red,
        "preview_b_red_count": b_red,
        "preview_difference": p_red - b_red,
        "preview_same_count": same,
        "preview_opposite_count": opposite,
    }
    row.update(derived_stats(eye, "eye"))
    row.update(derived_stats(small, "small"))
    row.update(derived_stats(roach, "roach"))
    for k, v in pprev.items():
        row[f"preview_p_{k}"] = v
    for k, v in bprev.items():
        row[f"preview_b_{k}"] = v
    for i in range(1, 21):
        row[f"lag_{i}"] = seq[-i] if len(seq) >= i else "N"
    return row


def _iter_compact_shoes():
    """Yield (shoe_id, sequence) one line at a time with constant memory use."""
    if not COMPACT_LIBRARY_PATH.exists():
        return
    with gzip.open(COMPACT_LIBRARY_PATH, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                sid, seq = line.split("\t", 1)
            else:
                sid, seq = "", line
            seq = clean_sequence(seq)
            if seq:
                yield sid, seq


def compact_library_score(seq: str) -> dict[str, Any]:
    """Search all compact shoes in one gzip pass without loading them into RAM."""
    seq = clean_sequence(seq)
    if len(seq) < LIBRARY_MIN_SUFFIX:
        raise HTTPException(400, f"Need at least {LIBRARY_MIN_SUFFIX} hands for library matching")
    if not COMPACT_LIBRARY_PATH.exists():
        raise HTTPException(503, f"Compact shoe library not found: {COMPACT_LIBRARY_PATH.name}")

    cache_key = seq[-LIBRARY_MAX_SUFFIX:]
    with _library_cache_lock:
        cached = _library_cache.get(cache_key)
        if cached is not None:
            _library_cache.move_to_end(cache_key)
            return dict(cached)

    started = time.perf_counter()
    max_len = min(LIBRARY_MAX_SUFFIX, len(seq))
    suffixes = {n: seq[-n:] for n in range(LIBRARY_MIN_SUFFIX, max_len + 1)}
    stats = {
        n: {"counts": {"P": 0, "B": 0, "T": 0}, "matched_shoes": 0}
        for n in suffixes
    }
    scanned = 0

    for _sid, hist in _iter_compact_shoes():
        scanned += 1
        for n, suffix in suffixes.items():
            shoe_hit = False
            start_at = 0
            while True:
                pos = hist.find(suffix, start_at)
                if pos < 0:
                    break
                nxt_idx = pos + n
                if nxt_idx < len(hist):
                    nxt = hist[nxt_idx]
                    if nxt in "PBT":
                        stats[n]["counts"][nxt] += 1
                        shoe_hit = True
                start_at = pos + 1
            if shoe_hit:
                stats[n]["matched_shoes"] += 1

    best = None
    fallback = None
    for n in range(max_len, LIBRARY_MIN_SUFFIX - 1, -1):
        counts = stats[n]["counts"]
        total = sum(counts.values())
        candidate = {
            "suffix": suffixes[n],
            "suffix_length": n,
            "matches": total,
            "matched_shoes": stats[n]["matched_shoes"],
            "scanned_shoes": scanned,
            "counts": counts,
        }
        if total > 0 and fallback is None:
            fallback = candidate
        if total >= LIBRARY_MIN_MATCHES:
            best = candidate
            break

    best = best or fallback or {
        "suffix": seq[-LIBRARY_MIN_SUFFIX:], "suffix_length": LIBRARY_MIN_SUFFIX,
        "matches": 0, "matched_shoes": 0, "scanned_shoes": scanned,
        "counts": {"P": 0, "B": 0, "T": 0},
    }
    total = best["matches"]
    if total:
        probs = {k: best["counts"][k] / total for k in ("P", "B", "T")}
    else:
        probs = {"P": 0.446, "B": 0.458, "T": 0.096}

    evidence = min(1.0, total / 500.0)
    spread = max(probs.values()) - min(probs.values())
    confidence = min(70.0, 20.0 + 40.0 * evidence + 25.0 * spread)
    result = {
        "p": probs["P"], "b": probs["B"], "t": probs["T"],
        "confidence": round(confidence, 1),
        **best,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "source": "compact-200k-stream",
    }
    with _library_cache_lock:
        _library_cache[cache_key] = dict(result)
        _library_cache.move_to_end(cache_key)
        while len(_library_cache) > LIBRARY_CACHE_SIZE:
            _library_cache.popitem(last=False)
    return result


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
    meta_path = APP_DIR / "latest.json"
    if not meta_path.exists():
        return
    try:
        ensure_h2o()
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
    r = pred.iloc[0]
    prob_cols = [c for c in pred.columns if c != "predict"]
    probs: dict[str, float] = {"P": 0.0, "B": 0.0, "T": 0.0}
    if len(prob_cols) == len(classes):
        for cls, col in zip(classes, prob_cols):
            probs[cls] = float(r[col])
    else:
        for cls in classes:
            if cls in pred.columns:
                probs[cls] = float(r[cls])
    total = sum(probs.values())
    if total <= 0:
        chosen = str(r.get("predict", "P"))
        probs[chosen] = 1.0
        total = 1.0
    return {k: v / total for k, v in probs.items()}



def _normalize_three_scores(p: Any, b: Any, t: Any = 0) -> dict[str, float]:
    vals = []
    for x in (p, b, t):
        try:
            vals.append(max(0.0, float(x)))
        except (TypeError, ValueError):
            vals.append(0.0)
    p, b, t = vals
    if max(p, b, t) > 1.0001:
        p, b, t = p / 100.0, b / 100.0, t / 100.0
    total = p + b + t
    if total <= 0:
        p, b, t, total = 0.5, 0.5, 0.0, 1.0
    return {"P": p / total, "B": b / total, "T": t / total}


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.I)
        text = re.sub(r"\\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\\{.*\\}", text, flags=re.S)
    if not match:
        raise ValueError("DeepSeek did not return a JSON object")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("DeepSeek JSON response was not an object")
    return obj


def call_deepseek_analysis(features: dict[str, Any], shoes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(503, "DEEPSEEK_API_KEY is not configured on the server")

    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    if len(seq) < 3:
        raise HTTPException(400, "Need at least 3 entered hands before DeepSeek analysis")
    row = features_for_sequence(seq)
    shoes = shoes or []

    # Only compact statistical summaries are sent. The API key remains server-side.
    historical = []
    suffix = seq[-8:]
    for shoe in shoes[-120:]:
        s = clean_sequence(str(shoe.get("sequence", "")))
        if not s:
            continue
        hit = s.rfind(suffix) if len(suffix) >= 3 else -1
        nxt = s[hit + len(suffix)] if hit >= 0 and hit + len(suffix) < len(s) else None
        historical.append({"length": len(s), "suffix_match_next": nxt})

    compact = {
        "sequence": seq,
        "hand_count": row["hand_count"],
        "player_count": row["player_count"],
        "banker_count": row["banker_count"],
        "tie_count": row["tie_count"],
        "pattern_type": row["pattern_type"],
        "streak_side": row["streak_side"],
        "streak_length": row["streak_length"],
        "alternations_20": row["alternations_20"],
        "big_road": {
            "columns": row["big_columns"],
            "last_col_depth": row["big_last_col_depth"],
            "prev_col_depth": row["big_prev_col_depth"],
            "max_depth": row["big_max_depth"],
        },
        "derived_roads": {
            "big_eye": {"last": row["eye_last"], "red_rate": row["eye_red_rate"], "recent_red_rate": row["eye_recent_red_rate"], "streak": row["eye_streak_length"]},
            "small": {"last": row["small_last"], "red_rate": row["small_red_rate"], "recent_red_rate": row["small_recent_red_rate"], "streak": row["small_streak_length"]},
            "roach": {"last": row["roach_last"], "red_rate": row["roach_red_rate"], "recent_red_rate": row["roach_recent_red_rate"], "streak": row["roach_streak_length"]},
            "preview_if_player": [row["preview_p_eye"], row["preview_p_small"], row["preview_p_roach"]],
            "preview_if_banker": [row["preview_b_eye"], row["preview_b_small"], row["preview_b_roach"]],
        },
        "saved_shoe_count": len(shoes),
        "historical_suffix_followups": historical,
        "current_shoe_color": features.get("current_shoe_color", ""),
    }

    system_msg = (
        "You are one component in a baccarat analysis dashboard. Analyze only the supplied observed history, "
        "road structure, and historical follow-up summaries. Do not claim certainty or a guaranteed edge. "
        "Return JSON only with keys p, b, t, confidence, pattern, rationale. p/b/t are comparative analysis "
        "scores totaling 100, confidence is 0-100 and should stay low when evidence conflicts or samples are small, "
        "pattern is a short label, and rationale is one concise sentence."
    )
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(compact, separators=(",", ":"))},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DEEPSEEK_TIMEOUT_SECS) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise HTTPException(exc.code, f"DeepSeek API error: {detail}")
    except Exception as exc:
        raise HTTPException(502, f"DeepSeek connection failed: {exc}")

    try:
        content = raw["choices"][0]["message"]["content"]
        out = _extract_json_object(content)
    except Exception as exc:
        raise HTTPException(502, f"Could not parse DeepSeek response: {exc}")
    probs = _normalize_three_scores(out.get("p"), out.get("b"), out.get("t", 0))
    try:
        conf = max(0.0, min(100.0, float(out.get("confidence", 50))))
    except (TypeError, ValueError):
        conf = 50.0
    return {
        "p": probs["P"], "b": probs["B"], "t": probs["T"],
        "confidence": conf,
        "pattern": str(out.get("pattern", row["pattern_type"]))[:80],
        "rationale": str(out.get("rationale", ""))[:500],
        "model": DEEPSEEK_MODEL,
        "sequence": seq,
        "road_snapshot": compact["derived_roads"],
    }


@app.on_event("startup")
def startup() -> None:
    load_latest()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "4.1-compact-200k-stream",
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL,
        "model_id": _model_id,
        "classes": _model_classes,
        "feature_count": len(FEATURE_COLUMNS),
        "compact_library": {
            "configured_path": str(COMPACT_LIBRARY_PATH),
            "available": COMPACT_LIBRARY_PATH.exists(),
            "size_bytes": COMPACT_LIBRARY_PATH.stat().st_size if COMPACT_LIBRARY_PATH.exists() else 0,
            "streaming": True,
        },
        "meta": _meta,
    }


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
            seed=20260810,
            sort_metric="logloss",
            balance_classes=True,
            exclude_algos=["DeepLearning"],
        )
        aml.train(x=FEATURE_COLUMNS, y="target", training_frame=hf_train, leaderboard_frame=hf_valid)
        leader = aml.leader
        model_path = h2o.save_model(leader, path=str(APP_DIR), force=True)
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
            "feature_version": "3.1-final-road-consensus",
            "feature_count": len(FEATURE_COLUMNS),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "training_shoes": int(len(unique_shoes) - len(valid_ids)),
            "validation_shoes": int(len(valid_ids)),
            "accuracy": accuracy,
            "leaderboard": aml.leaderboard.head(rows=5).as_data_frame().to_dict(orient="records"),
        }
        (APP_DIR / "latest.json").write_text(json.dumps(_meta, indent=2, default=str))
        return _meta


@app.post("/library-score")
def library_score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    features = payload.get("features") or payload
    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    return compact_library_score(seq)


@app.post("/score")
def score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Score with the full compact library; blend H2O only when a model exists."""
    auth(request)
    features = payload.get("features") or payload
    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    if len(seq) < MIN_PREFIX:
        raise HTTPException(400, f"Need at least {MIN_PREFIX} entered hands before scoring")
    row = features_for_sequence(seq)

    use_kaggle = bool(features.get("use_kaggle_library", payload.get("use_kaggle_library", True)))
    library = compact_library_score(seq) if use_kaggle else None
    final_probs = ({"P": library["p"], "B": library["b"], "T": library["t"]} if library else None)
    h2o_probs = None

    if _model is not None:
        try:
            ensure_h2o()
            hf = h2o.H2OFrame(pd.DataFrame([row], columns=FEATURE_COLUMNS))
            for col in CATEGORICAL:
                hf[col] = hf[col].asfactor()
            h2o_probs = prediction_probabilities(_model, hf, _model_classes)
            if use_kaggle and final_probs is not None:
                # Kaggle ON: blend the streamed 200k-shoe library with the saved-shoe H2O model.
                final_probs = {
                    k: 0.70 * final_probs[k] + 0.30 * h2o_probs[k]
                    for k in ("P", "B", "T")
                }
                total = sum(final_probs.values()) or 1.0
                final_probs = {k: v / total for k, v in final_probs.items()}
            else:
                # Kaggle OFF: use only the H2O model trained from the user's saved All In shoes.
                final_probs = dict(h2o_probs)
        except Exception as exc:
            # Do not make scoring fail just because H2O is unavailable on a
            # low-memory instance; the full streaming library still works.
            h2o_probs = {"error": str(exc)[:300]}

    if final_probs is None:
        raise HTTPException(409, "Kaggle shoe library is OFF and no trained H2O model is available. Train H2O from your saved All In Baccarat shoes first, or turn the Kaggle library back ON.")

    return {
        "p": final_probs["P"], "b": final_probs["B"], "t": final_probs["T"],
        "model_id": _model_id,
        "classes": _model_classes,
        "feature_version": "4.1-compact-200k-stream",
        "pattern_type": row["pattern_type"],
        "library": library,
        "kaggle_library_enabled": use_kaggle,
        "h2o": h2o_probs,
        "road_snapshot": {
            "eye": row["eye_last"], "small": row["small_last"], "roach": row["roach_last"],
            "preview_p": [row["preview_p_eye"], row["preview_p_small"], row["preview_p_roach"]],
            "preview_b": [row["preview_b_eye"], row["preview_b_small"], row["preview_b_roach"]],
        },
    }


@app.post("/deepseek-score")
def deepseek_score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    features = payload.get("features") or payload
    shoes = payload.get("shoes") or []
    return call_deepseek_analysis(features, shoes)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
