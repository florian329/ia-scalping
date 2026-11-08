"""Collecte de donnees reelles BTC via l'API publique Binance (sans cle API).

Poll les endpoints publics /api/v3/depth (20 niveaux) et /api/v3/aggTrades,
fusionne les deux flux en ticks au format du pipeline (timestamp, last_price,
last_volume, best_bid/ask, bid_size_i/ask_size_i) et sauvegarde un CSV.

Usage :
    python3 binance_collector.py --duration 120 --out btc_real_ticks.csv
"""

import argparse
import csv
import logging
import json
import time
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
DEPTH_LEVELS = 20


def _get(path, params=None, _retried=False):
    """Requete GET avec backoff sur rate-limit Binance (429/418).

    Binance indique le delai de reprise via l'en-tete Retry-After ; on le
    respecte puis on retente une fois. Les autres erreurs HTTP remontent.
    """
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "ia-scalping-collector/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 418) and not _retried:
            attente = float(exc.headers.get("Retry-After", 10))
            logger.warning("Rate limit Binance (%d) : pause %.0fs puis nouvel essai", exc.code, attente)
            time.sleep(attente)
            return _get(path, params, _retried=True)
        raise


def collect_tick(symbol, last_trade_id=None):
    """Un tick = snapshot du carnet + dernier trade agrege."""
    depth = _get("/api/v3/depth", {"symbol": symbol, "limit": DEPTH_LEVELS})
    trades = _get("/api/v3/aggTrades", {"symbol": symbol, "limit": 1})
    trade = trades[-1] if trades else None
    tick = {
        "timestamp": pd.to_datetime(trade["T"] if trade else time.time() * 1000, unit="ms", utc=True),
        "last_price": float(trade["p"]) if trade else float(depth["bids"][0][0]),
        "last_volume": float(trade["q"]) if trade else 0.0,
        "best_bid": float(depth["bids"][0][0]),
        "best_ask": float(depth["asks"][0][0]),
    }
    for i, (price, size) in enumerate(depth["bids"][:DEPTH_LEVELS], start=1):
        tick[f"bid_size_{i}"] = float(size)
    for i, (price, size) in enumerate(depth["asks"][:DEPTH_LEVELS], start=1):
        tick[f"ask_size_{i}"] = float(size)
    return tick


def collect(duration_seconds, symbol="BTCUSDT", poll_interval=0.25, verbose=True, ticks_out=None):
    """Collecte pendant duration_seconds ; ticks_out (liste) est alimente au fil
    de l'eau pour permettre une sauvegarde partielle en cas d'interruption."""
    ticks = ticks_out if ticks_out is not None else []
    n_errors = 0
    start = time.time()
    while time.time() - start < duration_seconds:
        try:
            ticks.append(collect_tick(symbol))
        except Exception as exc:
            n_errors += 1
            if verbose and n_errors % 10 == 1:
                logger.warning("erreur reseau (%s), reprise...", exc.__class__.__name__)
        time.sleep(poll_interval)
    if verbose:
        elapsed = time.time() - start
        print(
            f"Collecte terminee : {len(ticks)} ticks en {elapsed:.0f}s "
            f"({len(ticks) / elapsed:.1f} ticks/s, {n_errors} erreurs)"
        )
    return ticks


def save_csv(ticks, path):
    df = pd.DataFrame(ticks)
    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collecte de ticks reels Binance (public, sans cle)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration", type=int, default=120, help="Duree de collecte en secondes")
    parser.add_argument("--interval", type=float, default=0.25, help="Intervalle de polling en secondes")
    parser.add_argument("--out", default="btc_real_ticks.csv")
    args = parser.parse_args()

    print(f"Collecte de donnees reelles {args.symbol} pendant {args.duration}s...")
    ticks = collect(args.duration, symbol=args.symbol, poll_interval=args.interval)
    if not ticks:
        raise SystemExit("Aucun tick collecte.")
    df = save_csv(ticks, args.out)
    print(f"CSV sauvegarde : {args.out} ({len(df)} lignes)")
    print(f"Prix observe : {df['last_price'].min():.2f} - {df['last_price'].max():.2f} $")
    print(f"Spread moyen : {(df['best_ask'] - df['best_bid']).mean():.2f} $")


if __name__ == "__main__":
    main()
