# Defaults
DEFAULT_DAYS = 90
DEFAULT_COIN_ID = "bitcoin"
DEFAULT_VS_CURRENCY = "usd"

# RF hyperparams par défaut
RF_N_ESTIMATORS = 64
RF_THRESHOLD = 0.003
RF_FEE_RATE = 0.001

# Mode Auto-test (désactivé ici – l'app reste simple)
AUTO_TEST = False

# Param grid (utilisable si vous codez un mode autotest plus tard)
PARAM_GRID = {
    "method": ["rf", "ensemble"],
    "days": [60, 90],
    "n_lags": [12, 24],
    "horizon": [12, 18],
    "n_estimators": [32, 64, 128],
    "train_step": [1, 2],
    "threshold": [0.002, 0.003, 0.004],  # RF
    "trend_k": [0.5, 1.0, 1.5],
    "mr_k": [1.5, 2.0, 2.5],
    "vote_min": [2, 3],
    "fee_rate": [0.001],
}

RESULTS_DIR = "results"
SAVE_PER_RUN_DETAILS = False
