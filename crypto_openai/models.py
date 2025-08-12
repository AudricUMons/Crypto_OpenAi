import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor

def simulate_rf_follow(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    horizon: int = 6,
    n_lags: int = 12,
    train_step: int = 2,
    n_estimators: int = 64,
    fee_rate: float = 0.001,
    threshold: float = 0.002,
    progress_callback=None,
    sanity_break_alignment: bool = False,
    sanity_shuffle_target: bool = False,
    rng_seed: int = 0,
    return_trades: bool = False,
    return_signals: bool = False
):
    price = df['price'].to_numpy()
    ret   = df['return'].fillna(0).to_numpy()
    N = len(df)

    if N <= n_lags + horizon:
        out = pd.DataFrame(
            {
                'portfolio_value': [initial_cash]*N,
                'cash': [initial_cash]*N,
                'btc': [0.0]*N,
                'state': ['cash']*N,
            },
            index=df.index
        )
        if return_signals:
            out['pred'] = np.nan
            out['position'] = 0
        if return_trades:
            return out, ([], [], [], [])
        return out

    P = sliding_window_view(price, n_lags)
    R = sliding_window_view(ret,   n_lags)
    ma7  = df['MA7'].to_numpy()
    ma30 = df['MA30'].to_numpy()
    vol  = df['volatility'].to_numpy()
    tail_feats = np.column_stack([ma7[n_lags-1:], ma30[n_lags-1:], vol[n_lags-1:]])

    cash, btc, state = float(initial_cash), 0.0, "cash"
    values = [initial_cash]*n_lags
    cash_series = [initial_cash]*n_lags
    btc_series  = [0.0]*n_lags
    state_series = ["cash"]*n_lags

    pred_series = [np.nan]*n_lags
    pos_series  = [0]*n_lags

    buys_d, buys_p, sells_d, sells_p = [], [], [], []

    model = None
    total_steps = N - 1 - n_lags

    for count, i in enumerate(range(n_lags, N)):
        if progress_callback and total_steps > 0:
            progress_callback(int(100 * count / total_steps))

        rows = i - n_lags - horizon + 1
        if rows < 20:
            p = price[i]
            values.append(cash + btc * p)
            cash_series.append(cash)
            btc_series.append(btc)
            state_series.append(state)
            pred_series.append(np.nan)
            pos_series.append(1 if state == "btc" else 0)
            continue

        if ((i - n_lags) % train_step == 0) or (model is None):
            X_train = np.hstack([P[:rows], R[:rows], tail_feats[:rows]])
            start = n_lags + horizon - 1
            stop  = i
            if sanity_break_alignment:
                start -= 1; stop -= 1
            y_train = price[start:stop]
            if sanity_shuffle_target:
                y_train = y_train.copy()
                np.random.default_rng(rng_seed).shuffle(y_train)
            assert X_train.shape[0] == len(y_train)
            model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

        idx = i - n_lags
        X_pred = np.hstack([P[idx], R[idx], tail_feats[idx]])
        pred = model.predict([X_pred])[0]
        p = price[i]
        pct = (pred - p) / p

        if pct > threshold and state == "cash":
            buys_d.append(df.index[i]); buys_p.append(p)
            btc = (cash * (1 - fee_rate)) / p
            cash = 0.0
            state = "btc"
        elif pct < -threshold and state == "btc":
            sells_d.append(df.index[i]); sells_p.append(p)
            cash = btc * p * (1 - fee_rate)
            btc = 0.0
            state = "cash"

        values.append(cash + btc * p)
        cash_series.append(cash)
        btc_series.append(btc)
        state_series.append(state)
        pred_series.append(pred)
        pos_series.append(1 if state == "btc" else 0)

    while len(values) < N:
        values.append(values[-1]); cash_series.append(cash_series[-1]); btc_series.append(btc_series[-1]); state_series.append(state_series[-1])
        pred_series.append(np.nan); pos_series.append(pos_series[-1])

    out = pd.DataFrame({
        'portfolio_value': values,
        'cash': cash_series,
        'btc': btc_series,
        'state': state_series
    }, index=df.index)

    if return_signals:
        out['pred'] = pred_series
        out['position'] = pos_series

    if return_trades:
        return out, (buys_d, buys_p, sells_d, sells_p)
    return out

def simulate_three_signals_follow(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    horizon: int = 6,
    n_lags: int = 12,
    train_step: int = 2,
    n_estimators: int = 64,
    fee_rate: float = 0.001,
    threshold: float = 0.003,
    trend_k: float = 1.0,
    mr_k: float = 2.0,
    vote_min: int = 2,
    return_trades: bool = False,
    return_signals: bool = True
):
    pf_rf = simulate_rf_follow(
        df, initial_cash=initial_cash, horizon=horizon, n_lags=n_lags, train_step=train_step,
        n_estimators=n_estimators, fee_rate=fee_rate, threshold=threshold,
        return_trades=False, return_signals=True
    )
    price = df['price'].to_numpy()
    ma7   = df['MA7'].to_numpy()
    ma30  = df['MA30'].to_numpy()
    vol   = df['volatility'].to_numpy()
    pred  = pf_rf['pred'].to_numpy()

    N = len(df)
    cash, btc, state = float(initial_cash), 0.0, "cash"
    values = [initial_cash]*n_lags
    cash_series = [initial_cash]*n_lags
    btc_series  = [0.0]*n_lags
    state_series = ["cash"]*n_lags

    sig_rf_s = [0]*n_lags
    sig_tr_s = [0]*n_lags
    sig_mr_s = [0]*n_lags
    vote_s   = [0]*n_lags

    buys_d, buys_p, sells_d, sells_p = [], [], [], []

    for i in range(n_lags, N):
        p = price[i]

        s_rf = 0
        if not np.isnan(pred[i]):
            pct = (pred[i] - p) / p
            if pct > threshold: s_rf = +1
            elif pct < -threshold: s_rf = -1

        buf = trend_k * vol[i] * p
        s_tr = +1 if (ma7[i] - ma30[i]) > buf else (-1 if (ma30[i] - ma7[i]) > buf else 0)

        band = mr_k * vol[i] * p
        dev  = p - ma30[i]
        s_mr = +1 if dev < -band else (-1 if dev > band else 0)

        active = [s for s in (s_rf, s_tr, s_mr) if s != 0]
        v = np.sign(sum(active)) if len(active) >= vote_min else 0

        if v > 0 and state == "cash":
            buys_d.append(df.index[i]); buys_p.append(p)
            btc = (cash * (1 - fee_rate)) / p
            cash = 0.0; state = "btc"
        elif v < 0 and state == "btc":
            sells_d.append(df.index[i]); sells_p.append(p)
            cash = btc * p * (1 - fee_rate)
            btc = 0.0; state = "cash"

        values.append(cash + btc * p)
        cash_series.append(cash); btc_series.append(btc); state_series.append(state)
        sig_rf_s.append(s_rf); sig_tr_s.append(s_tr); sig_mr_s.append(s_mr); vote_s.append(v)

    while len(values) < N:
        values.append(values[-1]); cash_series.append(cash_series[-1]); btc_series.append(btc_series[-1]); state_series.append(state_series[-1])
        sig_rf_s.append(0); sig_tr_s.append(0); sig_mr_s.append(0); vote_s.append(0)

    out = pd.DataFrame({
        'portfolio_value': values,
        'cash': cash_series,
        'btc': btc_series,
        'state': state_series,
        'pred_rf': pred,
        'sig_rf': sig_rf_s,
        'sig_trend': sig_tr_s,
        'sig_meanrev': sig_mr_s,
        'vote': vote_s
    }, index=df.index)
    if return_trades:
        return out, (buys_d, buys_p, sells_d, sells_p)
    return out
