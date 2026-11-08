"""Feature engineering : variables du carnet d'ordres et du price action.

Corrections par rapport a la version initiale :
- OBI multi-profondeur parametrable (jusqu'a 20 niveaux si disponibles).
- Momentum calcule sur 10, 30 et 60 ticks (spec PDF) et non plus une seule fenetre.
- Ajout du bid-ask spread, de l'ecart au VWAP et de l'ATR ultra-court.
- Aucune fuite temporelle : chaque feature en t n'utilise que des donnees <= t.
"""

import numpy as np
import pandas as pd


def compute_order_book_imbalance(df, depth_levels=5):
    """OBI sur N niveaux : oscille entre -1 (vendeur) et +1 (acheteur)."""
    bid_volumes = np.zeros(len(df))
    ask_volumes = np.zeros(len(df))
    for i in range(1, depth_levels + 1):
        if f"bid_size_{i}" in df.columns and f"ask_size_{i}" in df.columns:
            bid_volumes += df[f"bid_size_{i}"].fillna(0).to_numpy()
            ask_volumes += df[f"ask_size_{i}"].fillna(0).to_numpy()
    return (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes + 1e-10)


def compute_bid_ask_spread(df):
    """Spread normalise par le prix, en points de base."""
    if "best_bid" in df.columns and "best_ask" in df.columns:
        spread = (df["best_ask"] - df["best_bid"]).to_numpy()
    else:
        mid = df["last_price"].to_numpy()
        spread = np.full(len(df), 0.0)
        for i in range(1, len(df)):
            if np.isfinite(mid[i - 1]) and mid[i] != mid[i - 1]:
                spread[i] = abs(mid[i] - mid[i - 1])
    return spread, spread / (df["last_price"].to_numpy() + 1e-10) * 10000.0


def compute_log_returns(prices, windows=(10, 30, 60)):
    """Rendements logarithmiques sur plusieurs fenetres de ticks."""
    out = {}
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    for w in windows:
        if w >= n:
            out[w] = np.zeros(n)
            continue
        shifted = np.concatenate((np.full(w, prices[0]), prices[:-w]))
        lr = np.log(prices / (shifted + 1e-10))
        lr[:w] = 0.0
        out[w] = lr
    return out


def compute_signed_volume_momentum(prices, volumes, window=10):
    """Momentum pondere par les volumes directionnes (tick rule)."""
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    price_diff = np.diff(prices, prepend=prices[0])
    trade_direction = np.sign(price_diff)
    signed_volumes = volumes * trade_direction
    return (
        pd.Series(signed_volumes)
        .rolling(window=window, min_periods=1)
        .sum()
        .to_numpy()
    )


def compute_vwap_deviation(prices, volumes):
    """Ecart relatif du prix au VWAP cumule de la session."""
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    cum_pv = np.cumsum(prices * volumes)
    cum_v = np.cumsum(volumes) + 1e-10
    vwap = cum_pv / cum_v
    return (prices - vwap) / (vwap + 1e-10)


def compute_tick_atr(prices, window=60):
    """ATR ultra-court approxime par la moyenne mobile du |rendement| en ticks."""
    prices = np.asarray(prices, dtype=np.float64)
    abs_ret = np.abs(np.diff(prices, prepend=prices[0]))
    atr = pd.Series(abs_ret).rolling(window=window, min_periods=1).mean().to_numpy()
    return atr


def compute_ofi(obi):
    """Order Flow Imbalance dynamique : variation de l'OBI entre t et t-1.
    L'OBI statique decrit un etat ; l'OFI capture l'acceleration du desequilibre,
    plus predictif du mouvement imminent."""
    obi = np.asarray(obi, dtype=np.float64)
    ofi = np.diff(obi, prepend=obi[0] if len(obi) else 0.0)
    return ofi


def compute_time_of_day(df):
    """Encodage cyclique de l'heure de session (sin/cos).
    Le comportement du marche a 14h30 n'a rien a voir avec 3h du matin ;
    l'encodage cyclique evite la discontinuite 23h59 -> 00h00."""
    n = len(df)
    if "timestamp" not in df.columns:
        return np.zeros(n), np.zeros(n)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour_fraction = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    angle = 2 * np.pi * hour_fraction / 24.0
    return np.sin(angle).to_numpy(), np.cos(angle).to_numpy()


def compute_vpin_rolling(prices, volumes, window=100):
    """VPIN simplifie (fenetre glissante) : fraction du volume agresseur vendeur.
    Proche de 1 = pression vendeuse extreme (flux toxique), proche de 0 =
    pression acheteuse. Detecte l'agressivite unidirectionnelle avant les
    grands mouvements."""
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    price_diff = np.diff(prices, prepend=prices[0])
    sell_volume = volumes * (price_diff < 0)
    roll_sell = pd.Series(sell_volume).rolling(window=window, min_periods=1).sum()
    roll_total = pd.Series(volumes).rolling(window=window, min_periods=1).sum()
    return (roll_sell / (roll_total + 1e-10)).to_numpy()


def build_features(df, depth_levels=5, momentum_windows=(10, 30, 60)):
    """Assemble toutes les features dans un DataFrame copie de l'entree."""
    out = df.copy().reset_index(drop=True)
    out["OBI"] = compute_order_book_imbalance(df, depth_levels=depth_levels)
    spread, spread_bps = compute_bid_ask_spread(df)
    out["Spread"] = spread
    out["Spread_BPS"] = spread_bps
    log_returns = compute_log_returns(df["last_price"], windows=momentum_windows)
    for w, lr in log_returns.items():
        out[f"Log_Return_{w}"] = lr
    out["VW_Momentum"] = compute_signed_volume_momentum(
        df["last_price"], df["last_volume"], window=10
    )
    out["VWAP_Dev"] = compute_vwap_deviation(df["last_price"], df["last_volume"])
    out["ATR_Tick"] = compute_tick_atr(df["last_price"], window=60)
    out["OFI"] = compute_ofi(out["OBI"])
    out["Hour_Sin"], out["Hour_Cos"] = compute_time_of_day(df)
    out["VPIN_Rolling"] = compute_vpin_rolling(df["last_price"], df["last_volume"], window=100)
    return out
