# Crypto_OpenAI

Une app Streamlit minimaliste pour tester trois méthodes de signal/trading intraday (1h) :
1) **RandomForest (RF)** — modèle ML local
2) **Ensemble 3 signaux** — RF + trend (MAs) + mean-reversion (vote 2/3)
3) **LLM** — décision ponctuelle BUY/SELL/HOLD via l'API OpenAI (Structured Outputs)

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Pour l'option **LLM**, définis la variable d'environnement :
```bash
export OPENAI_API_KEY="sk-..."
```

## Lancer l'app
```bash
streamlit run crypto_openai/app.py
```

## Notes
- Données 1h gratuites via Binance / CoinCap / CryptoCompare (sans clé).
- Les méthodes **RF** et **Ensemble** sont backtestées localement (all-in/out, frais).
- La méthode **LLM** fait **une décision ponctuelle** (pas de backtest complet).
- Ceci est un outil pédagogique — **pas un conseil financier**.
