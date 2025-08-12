
# --- make package imports work when run via `streamlit run crypto_openai/app.py`
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from crypto_openai.config import (
    DEFAULT_DAYS, DEFAULT_COIN_ID, DEFAULT_VS_CURRENCY,
    RF_N_ESTIMATORS, RF_THRESHOLD, RF_FEE_RATE
)
from crypto_openai.data import fetch_data
from crypto_openai.features import add_indicators
from crypto_openai.models import simulate_rf_follow, simulate_three_signals_follow
from crypto_openai.charts import render_main_chart, render_trades_chart
from crypto_openai.llm_method import decide_now_with_llm

@dataclass
class Options:
    coin_id: str = DEFAULT_COIN_ID
    vs_currency: str = DEFAULT_VS_CURRENCY
    days: int = DEFAULT_DAYS
    provider: str = "auto"
    method: str = "rf"  # 'rf', 'ensemble', 'llm'
    n_lags: int = 12
    horizon: int = 6
    train_step: int = 2
    n_estimators: int = RF_N_ESTIMATORS
    threshold: float = RF_THRESHOLD
    fee_rate: float = RF_FEE_RATE
    trend_k: float = 1.0
    mr_k: float = 2.0
    vote_min: int = 2
    llm_model: str = "gpt-5"
    min_edge: float = 0.003

def sidebar() -> Options:
    st.sidebar.title("⚙️ Paramètres")
    provider = st.sidebar.selectbox(
        "Source de données",
        ["auto", "binance", "coincap", "cryptocompare", "offline-sample"],
        index=0
    )
    coin_id = st.sidebar.selectbox("Coin", ["bitcoin","ethereum","solana","ripple","dogecoin","cardano","litecoin","polkadot","chainlink","tron"], index=0)
    vs_currency = st.sidebar.selectbox("Devise", ["usd","eur"], index=0)
    days = st.sidebar.slider("Jours d'historique", min_value=30, max_value=365, value=DEFAULT_DAYS, step=5)

    st.sidebar.markdown("---")
    method = st.sidebar.selectbox("Méthode", ["rf", "ensemble", "llm"], index=0)

    n_lags = st.sidebar.slider("n_lags", 6, 48, 12, 1)
    horizon = st.sidebar.slider("horizon (heures)", 3, 24, 6, 1)
    train_step = st.sidebar.slider("train_step", 1, 6, 2, 1)
    n_estimators = st.sidebar.select_slider("n_estimators", options=[16,32,64,128,256], value=RF_N_ESTIMATORS)
    threshold = st.sidebar.number_input("threshold (RF)", value=RF_THRESHOLD, step=0.001, format="%.3f")
    fee_rate = st.sidebar.number_input("fee_rate", value=RF_FEE_RATE, step=0.001, format="%.3f")

    trend_k = st.sidebar.number_input("trend_k (ensemble)", value=1.0, step=0.1, format="%.1f")
    mr_k = st.sidebar.number_input("mr_k (ensemble)", value=2.0, step=0.1, format="%.1f")
    vote_min = st.sidebar.select_slider("vote_min (ensemble)", options=[1,2,3], value=2)

    st.sidebar.markdown("---")
    llm_model = st.sidebar.text_input("LLM model", value="gpt-5")
    min_edge = st.sidebar.number_input("min_edge (LLM)", value=0.003, step=0.001, format="%.3f")

    return Options(
        provider=provider,
        coin_id=coin_id, vs_currency=vs_currency, days=days, method=method,
        n_lags=n_lags, horizon=horizon, train_step=train_step,
        n_estimators=n_estimators, threshold=float(threshold), fee_rate=float(fee_rate),
        trend_k=float(trend_k), mr_k=float(mr_k), vote_min=int(vote_min),
        llm_model=llm_model, min_edge=float(min_edge)
    )

def _run_rf(df: pd.DataFrame, opt: Options):
    pf, (buys_d, buys_p, sells_d, sells_p) = simulate_rf_follow(
        df,
        n_lags=opt.n_lags, horizon=opt.horizon, train_step=opt.train_step,
        n_estimators=opt.n_estimators, threshold=opt.threshold, fee_rate=opt.fee_rate,
        return_trades=True, return_signals=True
    )
    return pf, buys_d, buys_p, sells_d, sells_p

def _run_ensemble(df: pd.DataFrame, opt: Options):
    pf, (buys_d, buys_p, sells_d, sells_p) = simulate_three_signals_follow(
        df,
        n_lags=opt.n_lags, horizon=opt.horizon, train_step=opt.train_step,
        n_estimators=opt.n_estimators, threshold=opt.threshold, fee_rate=opt.fee_rate,
        trend_k=opt.trend_k, mr_k=opt.mr_k, vote_min=opt.vote_min,
        return_trades=True, return_signals=True
    )
    return pf, buys_d, buys_p, sells_d, sells_p

def load_data_with_feedback(opt) -> pd.DataFrame:
    # ordre conseillé : CoinCap → CryptoCompare → Binance
    providers: List[str] = (
        ["coincap", "cryptocompare", "binance"]
        if opt.provider == "auto" else [opt.provider]
    )
    with st.status("Chargement des données 1h…", expanded=True) as status:
        for p in providers:
            st.write(f"→ Essai provider: **{p}**")
            try:
                df = fetch_data(
                    days=opt.days,
                    coin_id=opt.coin_id,
                    vs_currency=opt.vs_currency,
                    provider=p
                )
                status.update(label=f"Données chargées via **{p}** ✅", state="complete")
                return df
            except Exception as e:
                st.write(f"❌ {p}: {e}")
        status.update(label="Échec du chargement depuis toutes les sources.", state="error")
        raise RuntimeError("Aucun provider n'a répondu")

def main():
    st.set_page_config(page_title="Crypto_OpenAI", layout="wide")
    st.title("🧠 Crypto_OpenAI — RF • Ensemble • LLM")

    opt = sidebar()

    with st.spinner(""):
        try:
            df = load_data_with_feedback(opt)  # uniquement vraies données
            df = add_indicators(df)
        except Exception as e:
            st.error(f"Impossible de charger les données : {e}")
            st.stop()


    if opt.method in ("rf", "ensemble"):
        if opt.method == "rf":
            pf, buys_d, buys_p, sells_d, sells_p = _run_rf(df, opt)
            series = [
                ("Buy & Hold", (df['price'] / df['price'].iloc[0]) * float(pf['portfolio_value'].iloc[0])),
                ("RandomForest", pf['portfolio_value']),
            ]
            st.plotly_chart(render_main_chart(df, series), use_container_width=True)
            st.plotly_chart(render_trades_chart(df, buys_d, buys_p, sells_d, sells_p, title="Trades (RF)"), use_container_width=True)
        else:
            pf, buys_d, buys_p, sells_d, sells_p = _run_ensemble(df, opt)
            series = [
                ("Buy & Hold", (df['price'] / df['price'].iloc[0]) * float(pf['portfolio_value'].iloc[0])),
                ("Ensemble (2/3)", pf['portfolio_value']),
            ]
            st.plotly_chart(render_main_chart(df, series), use_container_width=True)
            st.plotly_chart(render_trades_chart(df, buys_d, buys_p, sells_d, sells_p, title="Trades (Ensemble)"), use_container_width=True)

        last = pf.iloc[-1]
        price_now = float(df['price'].iloc[-1])
        cash_now = float(last.get('cash', 0.0))
        btc_now = float(last.get('btc', 0.0))
        state_now = str(last.get('state', 'cash'))

        if abs(cash_now) < 1e-8: cash_now = 0.0
        if abs(btc_now) < 1e-12: btc_now = 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("État", "CASH" if state_now == "cash" else "CRYPTO")
        col2.metric("Valeur portefeuille", f"${last['portfolio_value']:,.2f}")
        col3.metric("Cash", f"${cash_now:,.2f}")
        col4.metric("BTC", f"{btc_now:.6f} BTC (~${btc_now*price_now:,.2f})")

    else:
        st.info("La méthode LLM fait une **décision ponctuelle** BUY/SELL/HOLD sur la dernière bougie.")
        colL, colR = st.columns([2,1])
        with colL:
            st.plotly_chart(render_main_chart(df, [("Prix (normalisé)", (df['price']/df['price'].iloc[0]))]), use_container_width=True)
        with colR:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY n'est pas défini dans l'environnement.")
            if st.button("🧠 Demander au LLM maintenant"):
                res = decide_now_with_llm(df, symbol=f"{opt.coin_id.upper()}/{opt.vs_currency.upper()}", fee_rate=opt.fee_rate, min_edge=opt.min_edge, model=opt.llm_model)
                st.json(res)
            else:
                st.caption("Clique pour obtenir une décision LLM (JSON).")

    st.caption("⚠️ Outil pédagogique, non un conseil financier. Frais et slippage non exhaustifs.")

if __name__ == "__main__":
    main()
