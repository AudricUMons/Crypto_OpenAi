import os, json
from typing import Dict, Any
from datetime import datetime, timezone

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

_JSON_SCHEMA = {
    "name": "TradingSignal",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}
        },
        "required": ["action", "confidence"]
    },
    "strict": True
}

_SYSTEM = (
    "Tu es un moteur de signaux crypto intraday (bougies 1h). "
    "Décide STRICTEMENT parmi BUY, SELL ou HOLD, en te basant uniquement sur le snapshot fourni. "
    "Tiens compte des frais et d'un seuil minimal d'avantage utile (min_edge). "
    "Si l'espérance nette semble négative ou trop faible, choisis HOLD. "
    "Réponds uniquement en JSON conforme au schéma."
)

def _build_snapshot_from_df(df: pd.DataFrame, fee_rate: float, min_edge: float, symbol: str) -> Dict[str, Any]:
    last = df.iloc[-1]
    pred_rf = float(last["pred"]) if "pred" in df.columns and pd.notna(last["pred"]) else None
    snap = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": "unknown",
        "fee_rate": float(fee_rate),
        "min_edge": float(min_edge),
        "context": {"horizon_hours": 6, "n_lags": 12},
        "features": {
            "price": float(last["price"]),
            "MA7": float(last["MA7"]),
            "MA30": float(last["MA30"]),
            "volatility": float(last["volatility"]),
            "dist_ma7_ma30": float(last["MA7"] - last["MA30"]),
            "dist_price_ma30": float(last["price"] - last["MA30"]),
        }
    }
    if pred_rf is not None:
        snap["features"]["pred_rf"] = pred_rf
    return snap

def llm_signal(snapshot: Dict[str, Any], model: str = "gpt-5", temperature: float = 0.0, seed: int | None = None) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("La librairie openai n'est pas installée. Installez 'openai>=1.40'.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs = dict(
        model=model,
        temperature=temperature,
        seed=seed,
        response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
        input=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)},
        ],
    )
    try:
        if seed is not None:
            resp = client.responses.create(seed=seed, **kwargs)
        else:
            resp = client.responses.create(**kwargs)
    except TypeError:
        resp = client.responses.create(**kwargs)

    out_text = getattr(resp, "output_text", None)
    if not out_text:
        # fallback d’extraction au cas où
        out_text = "".join(
            getattr(c, "text", "")
            for item in getattr(resp, "output", [])
            for c in getattr(item, "content", [])
            if getattr(c, "type", "") == "output_text"
        )
    return json.loads(out_text)

def decide_now_with_llm(df: pd.DataFrame, symbol: str, fee_rate: float, min_edge: float, model: str = "gpt-5") -> Dict[str, Any]:
    snap = _build_snapshot_from_df(df, fee_rate=fee_rate, min_edge=min_edge, symbol=symbol)
    try:
        result = llm_signal(snap, model=model)
    except Exception as e:
        return {"action": "HOLD", "confidence": 0.0, "reason": f"Erreur LLM: {e}"}
    return result
