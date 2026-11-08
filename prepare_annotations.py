"""Preparation et validation du fichier d'annotations de trades.

Le fichier d'annotations est le carburant de l'apprentissage par imitation.
Une seule annotation erronee (timestamp mal converti, side invalide, trade
sans tick correspondant) degrade silencieusement l'entrainement. Ce module :

- genere un template CSV conforme ;
- valide un fichier existant : colonnes, timestamps UTC, cotes, prix,
  ordre temporel, doublons, couverture par les donnees de marche ;
- s'utilise en ligne de commande : python3 prepare_annotations.py --ann mes_trades.csv
"""

import argparse

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "asset", "side", "entry_price", "exit_price"]
OPTIONAL_COLUMNS = ["volume"]
EXAMPLE_ROWS = [
    {
        "timestamp": "2026-11-08T14:05:22.104Z",
        "asset": "NQ",
        "side": "BUY",
        "entry_price": 19850.25,
        "exit_price": 19854.75,
        "volume": 2,
    },
    {
        "timestamp": "2026-11-08T14:07:05.441Z",
        "asset": "BTC",
        "side": "SELL",
        "entry_price": 76420.50,
        "exit_price": 76395.00,
        "volume": 1.5,
    },
]


def generate_template(path="annotations_template.csv"):
    df = pd.DataFrame(EXAMPLE_ROWS)
    df.to_csv(path, index=False)
    print(f"Template genere : {path}")
    print("Format attendu : timestamp (ISO 8601 UTC), asset, side (BUY/SELL), entry_price, exit_price, volume (optionnel)")
    return path


def validate_annotations(path, market_csv=None, tick_tolerance_ms=1000):
    """Retourne un rapport de validation : erreurs (bloquantes) et avertissements."""
    errors, warnings = [], []
    try:
        ann = pd.read_csv(path)
    except Exception as exc:
        return {"valid": False, "errors": [f"Fichier illisible : {exc}"], "warnings": [], "n_trades": 0}

    missing = [c for c in REQUIRED_COLUMNS if c not in ann.columns]
    if missing:
        errors.append(f"Colonnes obligatoires manquantes : {missing}")
        return {"valid": False, "errors": errors, "warnings": warnings, "n_trades": 0}

    n = len(ann)
    if n == 0:
        errors.append("Aucune ligne d'annotation.")

    parsed = pd.to_datetime(ann["timestamp"], errors="coerce", utc=True)
    bad_ts = parsed.isna()
    if bad_ts.any():
        errors.append(f"{int(bad_ts.sum())} timestamp(s) non interpretables (attendu ISO 8601, ex : 2026-11-08T14:05:22.104Z)")
    ann["_ts"] = parsed

    sides = ann["side"].astype(str).str.upper().str.strip()
    bad_side = ~sides.isin(["BUY", "SELL"])
    if bad_side.any():
        errors.append(f"{int(bad_side.sum())} side invalide (attendu BUY ou SELL) : {sides[bad_side].unique()[:5].tolist()}")
    ann["_side"] = sides

    for col in ("entry_price", "exit_price"):
        prices = pd.to_numeric(ann[col], errors="coerce")
        bad_prices = prices.isna() | (prices <= 0)
        if bad_prices.any():
            errors.append(f"{int(bad_prices.sum())} {col} invalide (non numerique ou <= 0)")

    dup = ann.duplicated(subset=["timestamp", "asset", "side"])
    if dup.any():
        warnings.append(f"{int(dup.sum())} ligne(s) dupliquee(s) (timestamp+asset+side)")

    bad_order = parsed.diff().dt.total_seconds() < 0
    if bad_order.any():
        warnings.append(f"{int(bad_order.sum())} ligne(s) non triees chronologiquement")

    assets = ann["asset"].value_counts().to_dict()
    for asset, count in assets.items():
        if count < 30:
            warnings.append(
                f"Actif {asset} : seulement {count} trades — l'apprentissage par imitation a besoin d'un minimum de 30-50 trades par actif"
            )

    if market_csv is not None:
        try:
            mkt = pd.read_csv(market_csv)
            mkt["timestamp"] = pd.to_datetime(mkt["timestamp"], utc=True)
            ticks_int = mkt["timestamp"].astype("int64").to_numpy()
            ann_int = parsed.astype("int64").to_numpy()
            orphan = 0
            for i, ts_int in enumerate(ann_int):
                if pd.isna(parsed.iloc[i]):
                    continue
                pos = int(np.searchsorted(ticks_int, ts_int))
                near = ticks_int[max(0, pos - 50) : pos + 50]
                if len(near) == 0 or int(np.min(np.abs(near - ts_int))) > tick_tolerance_ms * 1_000_000:
                    orphan += 1
            if orphan:
                warnings.append(
                    f"{orphan} annotation(s) sans tick correspondant a +/-{tick_tolerance_ms} ms dans {market_csv}"
                )
        except Exception as exc:
            warnings.append(f"Verification des donnees de marche impossible : {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_trades": n,
        "assets": assets,
    }


def print_report(report):
    print("=" * 62)
    print(f"VALIDATION DES ANNOTATIONS — {report['n_trades']} trades")
    if report.get("assets"):
        print(f"Actifs : {report['assets']}")
    if report["errors"]:
        print("ERREURS (bloquantes) :")
        for e in report["errors"]:
            print(f"  [X] {e}")
    if report["warnings"]:
        print("AVERTISSEMENTS :")
        for w in report["warnings"]:
            print(f"  [!] {w}")
    if report["valid"] and not report["warnings"]:
        print("Fichier conforme, pret pour l'entrainement.")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="Validation du fichier d'annotations de trades")
    parser.add_argument("--ann", required=True, help="Chemin du CSV d'annotations")
    parser.add_argument("--market-csv", default=None, help="CSV de ticks pour verifier la couverture temporelle")
    parser.add_argument("--template", action="store_true", help="Generer un template au lieu de valider")
    parser.add_argument("--out", default="annotations_template.csv")
    args = parser.parse_args()

    if args.template:
        generate_template(args.out)
        return
    report = validate_annotations(args.ann, market_csv=args.market_csv)
    print_report(report)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
