"""Validation statistique : walk-forward, bootstrap, sensibilité aux coûts, régimes.

Répond aux biais restants du backtest classique :
- un seul split = une seule roulette -> walk-forward (n fenêtres successives) ;
- absence d'incertitude sur les métriques -> bootstrap (IC à 95 %) ;
- sensibilité aux coûts inconnue -> stress test du slippage ;
- régimes de marché confondus -> métriques segmentées.
"""

import copy

import numpy as np
import pandas as pd

from backtester import ASSET_COST_MODELS, run_backtest


def walk_forward_backtest(
    model,
    df,
    asset="BTC",
    n_folds=5,
    train_ratio=0.6,
    money_manager=None,
    sequence_length=20,
    norm_stats=None,
    filters=None,
    confidence_threshold=0.0,
    **backtest_kwargs,
):
    """Découpe df en n_folds fenêtres de test chronologiques et évalue le modèle
    sur chacune. Retourne la liste des rapports par fenêtre + un résumé."""
    n = len(df)
    start = int(n * train_ratio)
    fold_size = max(1, (n - start) // n_folds)
    reports = []
    for k in range(n_folds):
        lo = start + k * fold_size
        hi = lo + fold_size if k < n_folds - 1 else n
        window = df.iloc[max(0, lo - sequence_length) : hi].reset_index(drop=True)
        report = run_backtest(
            model,
            window,
            asset=asset,
            money_manager=money_manager,
            sequence_length=sequence_length,
            norm_stats=norm_stats,
            filters=filters,
            confidence_threshold=confidence_threshold,
            **backtest_kwargs,
        )
        report["fold"] = k + 1
        report["window"] = (lo, hi)
        reports.append(report)
    summary = {
        "n_folds": n_folds,
        "folds_profitables": sum(1 for r in reports if r["final_capital"] > r["initial_capital"]),
        "rendements_pct": [r["total_return_pct"] for r in reports],
        "expectancies": [r["expectancy_per_trade"] for r in reports],
        "max_drawdowns_pct": [r["max_drawdown_pct"] for r in reports],
        "n_trades": [r["n_trades"] for r in reports],
    }
    summary["rendement_moyen_pct"] = float(np.mean(summary["rendements_pct"]))
    return reports, summary


def bootstrap_metrics(trades, n_bootstrap=2000, ci=0.95, seed=42):
    """Intervalles de confiance bootstrap sur expectancy, win rate et profit factor.

    trades : liste d'objets Trade (backtester) ou liste de PnL (float).
    """
    if len(trades) == 0:
        return {
            "expectancy": (None, None),
            "win_rate": (None, None),
            "profit_factor": (None, None),
            "n_trades": 0,
            "ci": ci,
        }
    if hasattr(trades[0], "pnl_fill"):
        pnls = np.array(
            [getattr(t, "net_pnl", t.pnl_fill(t.exit_price)) for t in trades]
        )
    else:
        pnls = np.array(trades, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(pnls)
    boot_exp, boot_wr, boot_pf = [], [], []
    for _ in range(n_bootstrap):
        sample = pnls[rng.integers(0, n, size=n)]
        boot_exp.append(sample.mean())
        boot_wr.append((sample > 0).mean())
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        boot_pf.append(wins / losses if losses > 0 else np.inf)
    alpha = (1 - ci) / 2
    return {
        "expectancy": (float(np.quantile(boot_exp, alpha)), float(np.quantile(boot_exp, 1 - alpha))),
        "win_rate": (float(np.quantile(boot_wr, alpha)), float(np.quantile(boot_wr, 1 - alpha))),
        "profit_factor": (float(np.quantile(boot_pf, alpha)), float(np.quantile(boot_pf, 1 - alpha))),
        "expectancy_point": float(pnls.mean()),
        "win_rate_point": float((pnls > 0).mean()),
        "n_trades": n,
        "ci": ci,
    }


def run_cost_sensitivity(model, df, asset="BTC", slippage_multipliers=(0.5, 1.0, 2.0, 3.0), **backtest_kwargs):
    """Relance le backtest en faisant varier le slippage. Révèle le seuil de
    rupture au-delà duquel la stratégie cesse d'être rentable."""
    results = []
    base = ASSET_COST_MODELS.get(asset, ASSET_COST_MODELS["BTC"])
    for mult in slippage_multipliers:
        original = ASSET_COST_MODELS.get(asset)
        stressed = copy.deepcopy(base)
        stressed["slippage"] = base["slippage"] * mult
        ASSET_COST_MODELS[asset] = stressed
        try:
            report = run_backtest(model, df, asset=asset, **backtest_kwargs)
        finally:
            if original is not None:
                ASSET_COST_MODELS[asset] = original
        results.append(
            {
                "slippage_multiplier": mult,
                "slippage": stressed["slippage"],
                "total_return_pct": report["total_return_pct"],
                "n_trades": report["n_trades"],
                "expectancy_per_trade": report["expectancy_per_trade"],
                "total_fees": report["total_fees"],
            }
        )
    return results


def classify_regimes(df, atr_col="ATR_Tick", price_col="last_price", trend_window=100):
    """Classe chaque tick : tendance (hausse/baisse/range) x volatilité (haute/basse)."""
    prices = df[price_col].to_numpy(dtype=np.float64)
    atr = df[atr_col].to_numpy(dtype=np.float64) if atr_col in df else np.full(len(df), 1.0)
    trend = np.full(len(df), "range", dtype=object)
    if len(prices) > trend_window:
        slope = prices[trend_window:] - prices[:-trend_window]
        threshold = 2 * atr[trend_window:]
        trend[trend_window:] = np.where(
            slope > threshold, "hausse", np.where(slope < -threshold, "baisse", "range")
        )
    atr_threshold = float(np.median(atr))
    volatility = np.where(atr > atr_threshold, "haute", "basse")
    return trend, volatility


def metrics_by_regime(report, df, trend_window=100):
    """Métriques segmentées par régime de marché pour les trades du rapport."""
    trend, volatility = classify_regimes(df, trend_window=trend_window)
    out = {}
    for regime in ("hausse", "range", "baisse"):
        for vol in ("haute", "basse"):
            key = f"{regime}_vol_{vol}"
            pnls = []
            for tr in report["trades"]:
                idx = tr.entry_index
                if idx < len(trend) and trend[idx] == regime and volatility[idx] == vol:
                    pnls.append(tr.pnl_fill(tr.exit_price))
            arr = np.array(pnls) if pnls else np.array([])
            out[key] = {
                "n_trades": len(pnls),
                "win_rate": float((arr > 0).mean() * 100) if len(pnls) else None,
                "expectancy": float(arr.mean()) if len(pnls) else None,
            }
    return out


def print_validation_summary(summary, boot=None, cost_results=None, regimes=None):
    print("=" * 62)
    print("VALIDATION WALK-FORWARD")
    print(f"Fenetres profitables : {summary['folds_profitables']}/{summary['n_folds']}")
    for i, (ret, exp, dd) in enumerate(
        zip(summary["rendements_pct"], summary["expectancies"], summary["max_drawdowns_pct"])
    ):
        print(f"  Fenetre {i + 1}: rendement={ret:>8.2f} % | expectancy={exp:>8.2f} $ | maxDD={dd:>7.2f} %")
    print(f"Rendement moyen     : {summary['rendement_moyen_pct']:.2f} %")
    if boot:
        print("-" * 62)
        print(f"BOOTSTRAP ({boot['n_trades']} trades, IC {int(boot['ci'] * 100)} %)")
        if boot["n_trades"] == 0:
            print("Pas assez de trades pour estimer un intervalle de confiance.")
        else:
            print(f"Expectancy : {boot['expectancy_point']:>8.2f} $  IC [{boot['expectancy'][0]:.2f}, {boot['expectancy'][1]:.2f}]")
            print(f"Win rate   : {boot['win_rate_point'] * 100:>7.2f} %  IC [{boot['win_rate'][0] * 100:.2f}, {boot['win_rate'][1] * 100:.2f}]")
    if cost_results:
        print("-" * 62)
        print("SENSIBILITE AUX COUTS (slippage)")
        for r in cost_results:
            print(
                f"  x{r['slippage_multiplier']:<4} slippage={r['slippage']:<6.2f} "
                f"rendement={r['total_return_pct']:>8.2f} % | expectancy={r['expectancy_per_trade']:>7.2f} $"
            )
    if regimes:
        print("-" * 62)
        print("METRIQUES PAR REGIME")
        for key, m in regimes.items():
            if m["n_trades"]:
                print(
                    f"  {key:<18} trades={m['n_trades']:<4} win_rate={m['win_rate']:>6.2f} % "
                    f"expectancy={m['expectancy']:>7.2f} $"
                )
    print("=" * 62)
