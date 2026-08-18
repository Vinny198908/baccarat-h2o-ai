#!/usr/bin/env python3
"""H2O AutoML baccarat scoring server with roadmap/pattern features.

The browser app POSTs saved shoes to /train and the current sequence to /score.
The model uses only information available before the next hand; it cannot know future cards.
"""
from __future__ import annotations

import json
import os
import threading
import gzip
import time
from collections import OrderedDict
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
MAX_MODELS = int(os.environ.get("H2O_MAX_MODELS", "12"))
MAX_RUNTIME_SECS = int(os.environ.get("H2O_MAX_RUNTIME_SECS", "90"))
MIN_PREFIX = int(os.environ.get("BACCARAT_MIN_PREFIX", "6"))
COMPACT_LIBRARY_PATH = Path(os.environ.get("BACCARAT_COMPACT_LIBRARY", "./baccarat_shoes_compact.txt.gz")).resolve()
LIBRARY_MIN_SUFFIX = int(os.environ.get("BACCARAT_LIBRARY_MIN_SUFFIX", "4"))
LIBRARY_MAX_SUFFIX = int(os.environ.get("BACCARAT_LIBRARY_MAX_SUFFIX", "12"))
LIBRARY_MIN_MATCHES = int(os.environ.get("BACCARAT_LIBRARY_MIN_MATCHES", "30"))
LIBRARY_CACHE_SIZE = int(os.environ.get("BACCARAT_LIBRARY_CACHE_SIZE", "128"))
BOT_DECISION_BUDGET_SECS = min(10.0, float(os.environ.get("BACCARAT_BOT_DECISION_BUDGET_SECS", "10")))
BOT_STATE_PATH = APP_DIR / "ultimate_bot_state.json"

app = FastAPI(title="Baccarat H2O Ultimate Smart Controller", version="8.0")
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
_bot_lock = threading.Lock()
_bot_state: dict[str, Any] = {"version":"ultimate-v8-smart-controller","predictions":0,"resolved":0,"wins":0,"losses":0,"waits":0,"by_phase":{},"by_model":{},"pending":None,"recent":[]}


BACCARAT_RULES_VERSION = "punto-banco-rules-v1"
BACCARAT_RULES = {
    "variant": "Punto Banco / casino baccarat",
    "card_values": {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 0, "J": 0, "Q": 0, "K": 0,
    },
    "scoring": "Add card values and keep only the ones digit (modulo 10). Highest total wins; equal totals are a Tie.",
    "naturals": "If either initial two-card hand totals 8 or 9, both hands stand.",
    "player_rule": "Without a natural, Player draws on 0-5 and stands on 6-7.",
    "banker_rule_if_player_stands": "Without a natural, Banker draws on 0-5 and stands on 6-7.",
    "banker_rule_if_player_draws": {
        "0-2": "draw",
        "3": "draw unless Player third card is 8",
        "4": "draw if Player third card is 2-7",
        "5": "draw if Player third card is 4-7",
        "6": "draw if Player third card is 6-7",
        "7": "stand",
    },
    "dealing_order": "Two cards are dealt to Player and Banker; required third cards are then dealt by the fixed drawing rules.",
    "prediction_note": "Drawing rules determine how a hand is completed; they do not reveal the value of undealt cards or make the next hand outcome known in advance.",
}


def _card_value(card: Any) -> int:
    if isinstance(card, bool):
        raise ValueError("boolean is not a card")
    if isinstance(card, (int, float)) and int(card) == card:
        n = int(card)
        if n == 1:
            return 1
        if 2 <= n <= 9:
            return n
        if n in (10, 11, 12, 13):
            return 0
    s = str(card).strip().upper()
    aliases = {"ACE":"A", "JACK":"J", "QUEEN":"Q", "KING":"K", "T":"10"}
    s = aliases.get(s, s)
    if s in BACCARAT_RULES["card_values"]:
        return int(BACCARAT_RULES["card_values"][s])
    raise ValueError(f"Unsupported baccarat card: {card!r}")


def baccarat_total(cards: list[Any]) -> int:
    return sum(_card_value(c) for c in cards) % 10


def _banker_should_draw(banker_total: int, player_third_value: int | None, player_drew: bool) -> bool:
    if not player_drew:
        return banker_total <= 5
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return player_third_value != 8
    if banker_total == 4:
        return player_third_value is not None and 2 <= player_third_value <= 7
    if banker_total == 5:
        return player_third_value is not None and 4 <= player_third_value <= 7
    if banker_total == 6:
        return player_third_value is not None and 6 <= player_third_value <= 7
    return False


def baccarat_hand_state(player_cards: list[Any], banker_cards: list[Any]) -> dict[str, Any]:
    if not isinstance(player_cards, list) or not isinstance(banker_cards, list):
        raise HTTPException(400, "player_cards and banker_cards must be arrays")
    if len(player_cards) not in (2, 3) or len(banker_cards) not in (2, 3):
        raise HTTPException(400, "Each hand must currently contain 2 or 3 cards")
    try:
        pvals = [_card_value(c) for c in player_cards]
        bvals = [_card_value(c) for c in banker_cards]
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    p2 = sum(pvals[:2]) % 10
    b2 = sum(bvals[:2]) % 10
    natural = p2 in (8, 9) or b2 in (8, 9)
    player_should_draw = (not natural) and p2 <= 5
    player_drew = len(player_cards) == 3
    player_third = pvals[2] if player_drew else None

    legal = True
    issues: list[str] = []
    if natural and (player_drew or len(banker_cards) == 3):
        legal = False; issues.append("Natural 8/9: neither side may draw a third card.")
    if not natural:
        if player_should_draw and not player_drew:
            issues.append("Player is required to draw a third card before the hand is complete.")
        if (not player_should_draw) and player_drew:
            legal = False; issues.append("Player must stand on an initial total of 6 or 7.")

    banker_should_draw = False
    banker_rule_known = natural or (not player_should_draw) or player_drew
    if not natural and banker_rule_known:
        banker_should_draw = _banker_should_draw(b2, player_third, player_drew)
        banker_drew = len(banker_cards) == 3
        if banker_should_draw and not banker_drew:
            issues.append("Banker is required to draw a third card before the hand is complete.")
        if (not banker_should_draw) and banker_drew:
            legal = False; issues.append("Banker should stand under the fixed third-card table.")

    complete = legal and not any("required to draw" in x for x in issues)
    ptotal = sum(pvals) % 10
    btotal = sum(bvals) % 10
    result = None
    if complete:
        result = "P" if ptotal > btotal else "B" if btotal > ptotal else "T"

    next_action = "complete"
    if not legal:
        next_action = "invalid"
    elif natural:
        next_action = "complete"
    elif player_should_draw and not player_drew:
        next_action = "player_draw"
    elif banker_rule_known and banker_should_draw and len(banker_cards) == 2:
        next_action = "banker_draw"

    return {
        "rules_version": BACCARAT_RULES_VERSION,
        "legal": legal,
        "complete": complete,
        "next_action": next_action,
        "player_initial_total": p2,
        "banker_initial_total": b2,
        "player_total": ptotal,
        "banker_total": btotal,
        "natural": natural,
        "player_should_draw": player_should_draw,
        "banker_should_draw": banker_should_draw if banker_rule_known else None,
        "result": result,
        "issues": issues,
    }


@app.get("/baccarat/rules")
def baccarat_rules(request: Request) -> dict[str, Any]:
    auth(request)
    return {"knowledge_layer_live": True, "rules_version": BACCARAT_RULES_VERSION, "rules": BACCARAT_RULES}


@app.post("/baccarat/hand-state")
def baccarat_hand_state_endpoint(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    return baccarat_hand_state(payload.get("player_cards") or [], payload.get("banker_cards") or [])

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

    # Multi-window sequence evidence: do not rely on a single exact suffix.
    # Longer windows matter more; tiny samples are shrunk so they cannot dominate.
    mw = []
    agg = {"P": 0.0, "B": 0.0, "T": 0.0}
    agg_weight = 0.0
    agreeing_windows = 0
    usable_windows = 0
    leaders = []
    for n in range(LIBRARY_MIN_SUFFIX, max_len + 1):
        counts = stats[n]["counts"]
        mt = sum(counts.values())
        if mt <= 0:
            continue
        raw_probs = {k: counts[k] / mt for k in ("P", "B", "T")}
        sample = min(1.0, mt / 120.0)
        length_factor = (n / LIBRARY_MIN_SUFFIX) ** 1.35
        w = length_factor * (0.30 + 0.70 * sample)
        for k in agg:
            agg[k] += raw_probs[k] * w
        agg_weight += w
        leader = max(raw_probs, key=raw_probs.get)
        leaders.append(leader)
        usable_windows += 1
        mw.append({
            "length": n, "suffix": suffixes[n], "matches": mt,
            "matched_shoes": stats[n]["matched_shoes"],
            "counts": counts, "leader": leader,
            "p": round(raw_probs["P"], 5), "b": round(raw_probs["B"], 5), "t": round(raw_probs["T"], 5),
            "weight": round(w, 4),
        })

    if agg_weight:
        multi_probs = {k: agg[k] / agg_weight for k in agg}
        multi_leader = max(multi_probs, key=multi_probs.get)
        agreeing_windows = sum(1 for x in leaders if x == multi_leader)
        agreement = agreeing_windows / max(1, usable_windows)
        avg_sample = sum(min(1.0, x["matches"] / 120.0) for x in mw) / max(1, len(mw))
        longest = max((x["length"] for x in mw), default=0)
        length_quality = min(1.0, longest / max(8.0, float(LIBRARY_MAX_SUFFIX)))
        similarity_score = round(100 * (0.40 * agreement + 0.35 * avg_sample + 0.25 * length_quality))
    else:
        multi_probs = dict(probs)
        multi_leader = max(multi_probs, key=multi_probs.get)
        agreement = 0.0
        similarity_score = 0

    evidence = min(1.0, total / 500.0)
    spread = max(probs.values()) - min(probs.values())
    confidence = min(70.0, 20.0 + 40.0 * evidence + 25.0 * spread)
    result = {
        "p": probs["P"], "b": probs["B"], "t": probs["T"],
        "confidence": round(confidence, 1),
        **best,
        "multi_window": mw,
        "multi_p": round(multi_probs["P"], 5),
        "multi_b": round(multi_probs["B"], 5),
        "multi_t": round(multi_probs["T"], 5),
        "multi_leader": multi_leader,
        "window_agreement": round(agreement, 4),
        "similarity_score": similarity_score,
        "usable_windows": usable_windows,
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




@app.on_event("startup")
def startup() -> None:
    _load_bot_state()
    load_latest()


@app.get("/")
def home():
    index_path = Path(__file__).resolve().parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "8.0-smart-controller",
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
            "feature_version": "3.2-probe-engine-v2-compatible",
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



def _phase(hand_count: int) -> str:
    if hand_count >= 55: return "55+"
    if hand_count >= 40: return "40-54"
    if hand_count >= 16: return "16-39"
    return "1-15"


def _load_bot_state() -> None:
    global _bot_state
    try:
        if BOT_STATE_PATH.exists():
            saved = json.loads(BOT_STATE_PATH.read_text())
            if isinstance(saved, dict): _bot_state.update(saved)
    except Exception as exc:
        print("Could not load bot state:", exc)


def _save_bot_state() -> None:
    tmp = BOT_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_bot_state, indent=2, default=str))
    tmp.replace(BOT_STATE_PATH)


def _model_reliability(name: str, phase: str) -> float:
    stat = _bot_state.get("by_model", {}).get(name, {})
    ps = stat.get("by_phase", {}).get(phase, {})
    n = int(ps.get("resolved", 0)); w = int(ps.get("wins", 0))
    # Conservative shrinkage toward 50%; models must earn influence.
    return (w + 5.0) / (n + 10.0) if n else 0.5


def _road_vote(row: dict[str, Any]) -> dict[str, Any]:
    """Current-shoe Big/derived-road structural specialist."""
    p_marks = [row.get("preview_p_eye"), row.get("preview_p_small"), row.get("preview_p_roach")]
    b_marks = [row.get("preview_b_eye"), row.get("preview_b_small"), row.get("preview_b_roach")]
    current = [row.get("eye_last"), row.get("small_last"), row.get("roach_last")]
    weights = [1.0, .90, .80]
    p_score=b_score=used=0.0
    detail=[]
    for name,cur,pm,bm,w in zip(("eye","small","roach"),current,p_marks,b_marks,weights):
        if not cur or not pm or not bm or pm == bm:
            continue
        # Structural continuity only: red/blue marks are NOT Player/Banker labels.
        ps = 1.0 if pm == cur else -1.0
        bs = 1.0 if bm == cur else -1.0
        p_score += w*ps; b_score += w*bs; used += w
        detail.append(f"{name}:{cur}->{pm}/{bm}")
    if not used or abs(p_score-b_score) < .20:
        return {"name":"roads","side":None,"confidence":0.5,"evidence":"no discriminating current-road probe"}
    side="P" if p_score>b_score else "B"
    edge=min(1.0,abs(p_score-b_score)/(2*used))
    conf=min(.72,.52+.20*edge)
    return {"name":"roads","side":side,"confidence":conf,"evidence":"; ".join(detail)[:240]}


def _saved_library_vote(seq: str, saved_shoes: list[str]) -> dict[str, Any]:
    """Suffix-match specialist using only the selected saved-shoe library."""
    seq=clean_sequence(seq)
    shoes=[clean_sequence(x) for x in (saved_shoes or []) if len(clean_sequence(x))>=4]
    if len(seq)<3 or not shoes:
        return {"name":"library","side":None,"confidence":0.5,"evidence":"no selected saved-shoe evidence"}
    for k in range(min(14,len(seq)),2,-1):
        pat=seq[-k:]; p=b=t=total=0
        for hist in shoes:
            for i in range(0,max(0,len(hist)-k)):
                if hist[i:i+k]==pat:
                    nxt=hist[i+k]
                    if nxt=="P": p+=1
                    elif nxt=="B": b+=1
                    else: t+=1
                    total+=1
        if total>=3 or (k<=5 and total>=2):
            nt=p+b
            if nt<2: return {"name":"library","side":None,"confidence":0.5,"evidence":f"matches={total} suffix={k}"}
            side="P" if p>b else "B" if b>p else None
            if side is None: return {"name":"library","side":None,"confidence":0.5,"evidence":f"tie P={p} B={b} suffix={k}"}
            edge=abs(p-b)/nt
            # confidence is deliberately capped; history matches are not true probabilities
            conf=min(.82,.52+.22*edge+min(.06,total*.0025)+min(.04,k*.003))
            return {"name":"library","side":side,"confidence":conf,"evidence":f"matches={total} suffix={k} P={p} B={b}"}
    return {"name":"library","side":None,"confidence":0.5,"evidence":"no qualifying selected-library match"}


def _transition_vote(seq: str) -> dict[str, Any]:
    """Current-shoe regime specialist. Uses only transitions already observed in this shoe."""
    x=[c for c in clean_sequence(seq) if c in "PB"]
    if len(x)<12:
        return {"name":"regime","side":None,"confidence":0.5,"evidence":"need 12+ non-tie hands"}
    recent=x[-24:]
    last=recent[-1]
    same=sum(1 for a,b in zip(recent,recent[1:]) if a==b)
    flip=(len(recent)-1)-same
    # estimate P/B continuation separately to avoid a blanket 'follow streak' rule
    cont={"P":[0,0],"B":[0,0]}
    for a,b in zip(recent,recent[1:]):
        if a in cont:
            cont[a][0]+=1
            cont[a][1]+=int(a==b)
    n,k=cont[last]
    if n<4:
        return {"name":"regime","side":None,"confidence":0.5,"evidence":f"insufficient {last} transition sample"}
    rate=(k+2)/(n+4)  # shrink toward .5
    if abs(rate-.5)<.08:
        return {"name":"regime","side":None,"confidence":0.5,"evidence":f"mixed continuation={rate:.2f}"}
    side=last if rate>.5 else ("B" if last=="P" else "P")
    conf=min(.66,.51+abs(rate-.5)*.38)
    return {"name":"regime","side":side,"confidence":conf,"evidence":f"{last} continuation={rate:.2f}; same/flip={same}/{flip}"}


def _late_shoe_vote(seq: str, saved_shoes: list[str]) -> dict[str, Any]:
    """40+ hand specialist. Compares only late-shoe states in saved shoes."""
    x=clean_sequence(seq)
    if len(x)<40:
        return {"name":"late_shoe","side":None,"confidence":0.5,"evidence":"inactive before hand 40"}
    shoes=[clean_sequence(z) for z in (saved_shoes or []) if len(clean_sequence(z))>=45]
    if not shoes:
        return {"name":"late_shoe","side":None,"confidence":0.5,"evidence":"no long saved shoes"}
    for k in (10,8,6,5,4):
        pat=x[-k:]; p=b=total=0
        for hist in shoes:
            # only compare occurrences at/after hand 35 to specialize on late shoe
            for i in range(max(35,k),len(hist)-k):
                if hist[i:i+k]==pat:
                    nxt=hist[i+k]
                    if nxt=="P": p+=1
                    elif nxt=="B": b+=1
                    total+=1
        if total>=3 and p!=b:
            side="P" if p>b else "B"; edge=abs(p-b)/(p+b)
            conf=min(.76,.52+.20*edge+min(.04,total*.002))
            return {"name":"late_shoe","side":side,"confidence":conf,"evidence":f"late matches={total} suffix={k} P={p} B={b}"}
    return {"name":"late_shoe","side":None,"confidence":0.5,"evidence":"no qualifying late-shoe match"}


def _ultimate_decision(seq: str, saved_shoes: list[str] | None=None) -> dict[str, Any]:
    """Adaptive controller over independent specialists with persistent reliability."""
    started=time.perf_counter(); row=features_for_sequence(seq); phase=_phase(len(seq)); saved=saved_shoes or []
    votes=[_road_vote(row),_saved_library_vote(seq,saved),_transition_vote(seq),_late_shoe_vote(seq,saved)]
    scores={"P":0.0,"B":0.0}; active=0; active_sides=[]
    for v in votes:
        side=v.get("side")
        if side not in scores: continue
        rel=_model_reliability(v["name"],phase); v["reliability"]=round(rel,4)
        signal=max(0.0,min(.50,(float(v.get("confidence",.5))-.5)*2.0))
        # reliability shrinkage: poor specialists lose influence, but new ones still get a fair trial
        rel_mult=.55+max(.25,min(.75,rel))
        weight=max(.01,signal*rel_mult)
        if v["name"]=="late_shoe" and phase in ("40-54","55+"): weight*=1.15
        v["weight"]=round(weight,4)
        scores[side]+=weight; active+=1; active_sides.append(side)
    total=scores["P"]+scores["B"]
    leader=max(scores,key=scores.get) if total else None
    agreement=(scores[leader]/total) if total and leader else .5
    vote_agreement=(active_sides.count(leader)/active) if active and leader else 0
    elapsed=time.perf_counter()-started
    # Conservative gate: never force a pick from one weak specialist.
    min_active=2
    min_weight_agreement=.64 if phase in ("40-54","55+") else .62
    min_vote_agreement=.60
    decision=leader if (leader and active>=min_active and agreement>=min_weight_agreement and vote_agreement>=min_vote_agreement and elapsed<=BOT_DECISION_BUDGET_SECS) else "WAIT"
    # This is controller agreement, not a true next-hand probability.
    confidence=agreement if decision!="WAIT" else .5
    return {"decision":decision,"confidence":round(confidence,4),"confidence_type":"controller_agreement_not_win_probability",
            "phase":phase,"hand_count":len(seq),"decision_seconds":round(elapsed,4),"budget_seconds":BOT_DECISION_BUDGET_SECS,
            "models_complete":len(votes),"models_participating":active,"vote_agreement":round(vote_agreement,4),"votes":votes,"scores":scores,
            "library":{"saved_shoes_received":len(saved)},"architecture":"adaptive specialists: current roads + selected saved library + current-shoe regime + 40+ late-shoe",
            "rules_knowledge":{"live":True,"version":BACCARAT_RULES_VERSION,"role":"game rules / hand validation only; not an extra prediction vote"}}


@app.post("/bot/decide")
def bot_decide(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request); features=payload.get("features") or payload
    seq=clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    if len(seq)<MIN_PREFIX: raise HTTPException(400,f"Need at least {MIN_PREFIX} entered hands before scoring")
    library_mode=str(features.get("library_mode",payload.get("library_mode","selected"))).lower()
    if library_mode not in ("selected","combined"): library_mode="selected"
    shoe_color=str(features.get("shoe_color",payload.get("shoe_color",""))).lower() or None
    raw_saved=features.get("saved_shoes") or payload.get("saved_shoes") or []
    saved_shoes=[str(x.get("sequence","") if isinstance(x,dict) else x) for x in raw_saved][:5000]
    result=_ultimate_decision(seq,saved_shoes)
    result["library_mode"]=library_mode; result["shoe_color"]=shoe_color
    prediction_id=f"{int(time.time()*1000)}-{abs(hash(seq))%1000000}"
    result["prediction_id"]=prediction_id if result["decision"]!="WAIT" else None
    with _bot_lock:
        _bot_state["predictions"]=int(_bot_state.get("predictions",0))+1
        if result["decision"]=="WAIT":
            _bot_state["waits"]=int(_bot_state.get("waits",0))+1; _bot_state["pending"]=None
        else:
            _bot_state["pending"]={"prediction_id":prediction_id,"sequence":seq,"prediction":result["decision"],"phase":result["phase"],"votes":result["votes"],"created_at":time.time(),"library_mode":library_mode,"shoe_color":shoe_color}
        _save_bot_state()
    result["performance"]={k:_bot_state.get(k,0) for k in ("predictions","resolved","wins","losses","waits")}
    return result


@app.post("/bot/feedback")
def bot_feedback(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request); actual=clean_sequence(str(payload.get("actual") or ""))[:1]
    if actual not in "PBT": raise HTTPException(400,"actual must be P, B, or T")
    with _bot_lock:
        pending=_bot_state.get("pending")
        if not pending: return {"updated":False,"reason":"no pending prediction","performance":_bot_state}
        supplied=payload.get("prediction_id")
        if supplied and pending.get("prediction_id") and supplied!=pending.get("prediction_id"):
            return {"updated":False,"reason":"stale prediction id","performance":_bot_state}
        pred=pending.get("prediction"); phase=pending.get("phase","unknown"); win=(actual==pred)
        _bot_state["resolved"]=int(_bot_state.get("resolved",0))+1
        _bot_state["wins" if win else "losses"]=int(_bot_state.get("wins" if win else "losses",0))+1
        ps=_bot_state.setdefault("by_phase",{}).setdefault(phase,{"resolved":0,"wins":0,"losses":0}); ps["resolved"]+=1; ps["wins" if win else "losses"]+=1
        for v in pending.get("votes",[]):
            side=v.get("side"); name=v.get("name")
            if not name or side not in "PB": continue
            ms=_bot_state.setdefault("by_model",{}).setdefault(name,{"resolved":0,"wins":0,"losses":0,"by_phase":{}}); ms["resolved"]+=1
            good=(side==actual); ms["wins" if good else "losses"]+=1
            mps=ms["by_phase"].setdefault(phase,{"resolved":0,"wins":0,"losses":0}); mps["resolved"]+=1; mps["wins" if good else "losses"]+=1
        recent=_bot_state.setdefault("recent",[]); recent.append({"prediction":pred,"actual":actual,"win":win,"phase":phase,"ts":time.time()}); _bot_state["recent"]=recent[-500:]
        _bot_state["pending"]=None; _save_bot_state()
        return {"updated":True,"win":win,"prediction":pred,"actual":actual,"performance":_bot_state}


# Backward-compatible aliases so older frontends do not 404.
@app.post("/feedback")
def legacy_feedback(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    return bot_feedback(payload, request)

@app.post("/ultimate-score")
def legacy_ultimate_score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    return bot_decide(payload, request)


@app.get("/bot/status")
def bot_status(request: Request) -> dict[str, Any]:
    auth(request); resolved=int(_bot_state.get("resolved",0)); wins=int(_bot_state.get("wins",0))
    return {"live":True,"version":"ultimate-v8-smart-controller","decision_budget_seconds":BOT_DECISION_BUDGET_SECS,"accuracy":round(wins/resolved,4) if resolved else None,"state":_bot_state,"model_id":_model_id,"rules_knowledge":{"live":True,"version":BACCARAT_RULES_VERSION}}


@app.post("/library-score")
def library_score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    auth(request)
    features = payload.get("features") or payload
    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    return compact_library_score(seq)


@app.post("/score")
def score(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Score ONLY with the trained H2O model. Kaggle/library scoring is isolated at /library-score."""
    auth(request)
    features = payload.get("features") or payload
    seq = clean_sequence(str(features.get("sequence") or features.get("recent_sequence") or ""))
    if len(seq) < MIN_PREFIX:
        raise HTTPException(400, f"Need at least {MIN_PREFIX} entered hands before scoring")
    if _model is None:
        raise HTTPException(409, "No trained H2O model is loaded. Train/retrain H2O from your saved shoes first.")

    row = features_for_sequence(seq)
    try:
        ensure_h2o()
        hf = h2o.H2OFrame(pd.DataFrame([row], columns=FEATURE_COLUMNS))
        for col in CATEGORICAL:
            hf[col] = hf[col].asfactor()
        h2o_probs = prediction_probabilities(_model, hf, _model_classes)
    except Exception as exc:
        raise HTTPException(502, f"H2O scoring failed: {str(exc)[:400]}")

    return {
        "p": h2o_probs["P"], "b": h2o_probs["B"], "t": h2o_probs["T"],
        "model_id": _model_id,
        "classes": _model_classes,
        "feature_version": "7.0-h2o-isolated",
        "pattern_type": row["pattern_type"],
        "source": "h2o_saved_shoe_model_only",
        "kaggle_used": False,
        "road_snapshot": {
            "eye": row["eye_last"], "small": row["small_last"], "roach": row["roach_last"],
            "preview_p": [row["preview_p_eye"], row["preview_p_small"], row["preview_p_roach"]],
            "preview_b": [row["preview_b_eye"], row["preview_b_small"], row["preview_b_roach"]],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
