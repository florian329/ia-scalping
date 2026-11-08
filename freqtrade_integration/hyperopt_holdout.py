"""Hyperopt avec HOLDOUT STRICT pour la strategie IaScalpingStrategy.

Le piege classique de l'hyperopt : optimiser sur tout l'historique puis
s'etonner que le marche "ait change". Ce script impose la discipline :

1. PHASE TRAIN  : hyperopt sur la periode d'entrainement uniquement ;
2. EXPORT       : les meilleurs parametres sont exportes vers
                  user_data/strategies/IaScalpingStrategy.json ;
3. PHASE HOLDOUT : backtest sur une periode JAMAIS touche par l'hyperopt ;
4. VERDICT      : comparaison train vs holdout — si les performances
                  s'effondrent sur le holdout, les parametres sont surajustes
                  et il ne faut PAS les utiliser en live.

Usage :
    python freqtrade_integration/hyperopt_holdout.py \
        --train 20261105-20261107 --holdout 20261107- --epochs 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FREQTRADE_CMD = [sys.executable, "-m", "freqtrade"]
BASE_ARGS = [
    "--config",
    "freqtrade_integration/config_ia_scalping.json",
    "--userdir",
    "user_data",
    "--strategy-path",
    "freqtrade_integration",
    "--strategy",
    "IaScalpingStrategy",
]


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"Echec de la commande (code {result.returncode})")
    return result.stdout


def parse_backtest_summary(output: str) -> dict:
    if "No trades made" in output:
        return {"trades": 0, "profit_pct": 0.0, "profit_factor": 0.0}

    def extract(label, cast=str):
        for line in output.splitlines():
            if label in line:
                for part in line.split("│"):
                    part = part.strip()
                    if part and part != label and not part.startswith(label):
                        try:
                            return cast(part.replace("%", "").strip())
                        except ValueError:
                            continue
        return None

    return {
        "trades": extract("Total/Daily Avg Trades", lambda x: int(x.split("/")[0].strip())),
        "profit_pct": extract("Total profit %", float),
        "profit_factor": extract("Profit factor", float),
        "expectancy": extract("Expectancy (Ratio)", str),
        "drawdown": extract("Absolute Drawdown", str),
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperopt avec holdout strict")
    parser.add_argument("--train", required=True, help="Timerange d'entrainement, ex: 20261105-20261107")
    parser.add_argument("--holdout", required=True, help="Timerange holdout JAMAIS vue par l'hyperopt, ex: 20261107-")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--spaces", default="entry", help="Espaces: entry, exit, roi, stoploss, protections")
    parser.add_argument("--jobs", type=int, default=1, help="Jobs paralleles (1 = sur, multiprocess fragile avec torch)")
    args = parser.parse_args()

    print("=" * 62)
    print("PHASE 1/4 : HYPEROPT (periode d'entrainement uniquement)")
    print("=" * 62)
    run(
        FREQTRADE_CMD
        + [
            "hyperopt",
            *BASE_ARGS,
            "--hyperopt-loss",
            "SharpeHyperOptLoss",
            "--spaces",
            args.spaces,
            "--timerange",
            args.train,
            "-e",
            str(args.epochs),
            "--random-state",
            "42",
            "-j",
            str(args.jobs),
        ]
    )

    print("=" * 62)
    print("PHASE 2/4 : EXPORT DES MEILLEURS PARAMETRES")
    print("=" * 62)
    results_dir = REPO_ROOT / "user_data" / "hyperopt_results"
    result_files = sorted(results_dir.glob("strategy_IaScalpingStrategy_*.fthypt"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise SystemExit("Aucun fichier de resultats hyperopt trouve.")
    latest = result_files[-1].name
    show_args = [
        "--config",
        "freqtrade_integration/config_ia_scalping.json",
        "--userdir",
        "user_data",
        "--hyperopt-filename",
        latest,
        "--best",
        "--print-json",
    ]
    best_json = run(FREQTRADE_CMD + ["hyperopt-show", *show_args])
    if not best_json.strip():
        print("Hyperopt sans bon resultat (aucun trade profitable en train).")
        print("Le hyperopt-loss Sharpe exige des trades ; essayez plus d'epochs ou")
        print("une loss plus permissive (ex: CalmarHyperOptLoss) ou un meilleur modele.")
        raise SystemExit(1)
    params_path = Path("user_data/strategies")
    params_path.mkdir(parents=True, exist_ok=True)
    out_file = params_path / "IaScalpingStrategy.json"
    with open(REPO_ROOT / out_file, "w") as f:
        f.write(best_json)
    print(f"Parametres sauvegardes : {out_file}")
    try:
        params = json.loads(best_json)
        print(json.dumps(params.get("params", params), indent=2)[:800])
    except json.JSONDecodeError:
        print(best_json[:800])

    print("=" * 62)
    print("PHASE 3/4 : BACKTEST HOLDOUT (jamais vue par l'hyperopt)")
    print("=" * 62)
    holdout_output = run(
        FREQTRADE_CMD
        + ["backtesting", *BASE_ARGS, "--timerange", args.holdout, "--breakdown", "day"]
    )
    train_output = run(
        FREQTRADE_CMD + ["backtesting", *BASE_ARGS, "--timerange", args.train]
    )

    print("=" * 62)
    print("PHASE 4/4 : VERDICT TRAIN vs HOLDOUT")
    print("=" * 62)
    train_stats = parse_backtest_summary(train_output)
    holdout_stats = parse_backtest_summary(holdout_output)

    print(f"{'Metrique':<22}{'Train':>15}{'Holdout':>15}")
    for key in ("trades", "profit_pct", "profit_factor"):
        print(f"{key:<22}{str(train_stats[key]):>15}{str(holdout_stats[key]):>15}")

    train_profit = train_stats["profit_pct"]
    holdout_profit = holdout_stats["profit_pct"]
    if train_profit is not None and holdout_profit is not None:
        if train_profit > 0 and holdout_profit > 0:
            print("\nVerdict : profitable en train ET en holdout -> parametres utilisables.")
        elif train_profit > 0 > holdout_profit:
            print("\nVerdict : SURAJUSTEMENT — profitable en train, perdant en holdout.")
            print("N'utilisez PAS ces parametres en live.")
        else:
            print("\nVerdict : non rentable meme en train -> le signal n'est pas la,")
            print("les parametres n'y changeront rien (voir annotations).")
    print("\nRappel : le holdout ne doit etre regarde qu'une fois par jeu de parametres.")
    print("Iterer sur le holdout = se mentir a soi-meme.")


if __name__ == "__main__":
    main()
