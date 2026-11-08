"""Collecte longue de donnees BTC : segmentation, reprise et concatenation.

Corrige les limites d'une collecte de plusieurs heures en un seul processus :
- segments de 7.5 min sauvegardes immediatement (aucune perte si interruption) ;
- retries des segments vides (reseau instable) avec arret propre au-dela de 5
  echecs consecutifs ;
- sauvegarde partielle sur Ctrl-C (les ticks du segment en cours ne sont pas perdus) ;
- validation des segments a la reprise (fichier tronque = recollecte) ;
- detection des trous temporels et dedoublonnage a la concatenation ;
- backoff sur rate-limit Binance gere dans binance_collector.

Exemples :
    python3 collect_long.py --hours 24 --out btc_24h.csv
    python3 collect_long.py --hours 24 --out btc_24h.csv --resume
    caffeinate -is nohup python3 collect_long.py --hours 24 --out btc_24h.csv \\
        > collecte.log 2>&1 &
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from binance_collector import collect, save_csv

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_SECONDS = 450
MAX_SEGMENT_RETRIES = 3
RETRY_DELAY_SECONDS = 30.0
MAX_CONSECUTIVE_FAILURES = 5
GAP_WARNING_SECONDS = 600.0


def segments_dir(out_path: Path) -> Path:
    return out_path.parent / (out_path.stem + "_segments")


def existing_segments(directory: Path, symbol: str) -> list:
    if not directory.exists():
        return []
    return sorted(
        directory.glob(f"seg_{symbol}_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )


def segment_valide(path: Path) -> bool:
    """Un segment est exploitable s'il est lisible et contient des ticks."""
    try:
        df = pd.read_csv(path)
        return len(df) > 0 and "last_price" in df.columns
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return False


def collect_segment_with_retries(
    segment_seconds: float,
    symbol: str,
    poll_interval: float,
    max_retries: int = MAX_SEGMENT_RETRIES,
    retry_delay: float = RETRY_DELAY_SECONDS,
    collect_fn=collect,
    sleep_fn=time.sleep,
    ticks_out=None,
):
    """Tente de collecter un segment ; reessaie si le reseau ne donne rien.

    Retourne (ticks, buffer_partiel) — le buffer permet la sauvegarde
    partielle sur interruption meme apres des echecs intermediaires.
    """
    buffer = ticks_out if ticks_out is not None else []
    for essai in range(1, max_retries + 1):
        try:
            ticks = collect_fn(
                segment_seconds,
                symbol=symbol,
                poll_interval=poll_interval,
                verbose=False,
                ticks_out=buffer,
            )
        except KeyboardInterrupt:
            raise
        if ticks:
            return ticks, buffer
        logger.warning(
            "segment vide (essai %d/%d) — reseau indisponible ?",
            essai,
            max_retries,
        )
        if essai < max_retries:
            sleep_fn(retry_delay)
    return [], buffer


def verifier_continuite(df: pd.DataFrame, gap_max_s: float = GAP_WARNING_SECONDS) -> list:
    """Detecte les trous temporels dans le flux concatene."""
    ecarts = df["timestamp"].diff().dt.total_seconds().dropna()
    trous = ecarts[ecarts > gap_max_s]
    if len(trous):
        logger.warning(
            "%d trou(s) temporel(s) : plus grand = %.1f min (seuil %.0f s)",
            len(trous),
            trous.max() / 60.0,
            gap_max_s,
        )
    return trous.index.tolist()


def concatenate(directory: Path, symbol: str, out_path: Path) -> pd.DataFrame:
    files = existing_segments(directory, symbol)
    if not files:
        raise SystemExit("Aucun segment collecte.")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    avant = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    doublons = avant - len(df)
    trous = verifier_continuite(df)
    df.to_csv(out_path, index=False)
    duration_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    print("=" * 62)
    print(f"CSV final : {out_path}")
    print(f"  {len(df)} ticks | {duration_min:.1f} min couvertes | {len(files)} segments")
    print(f"  prix {df['last_price'].min():.2f} - {df['last_price'].max():.2f} $")
    if doublons:
        print(f"  {doublons} doublon(s) supprime(s)")
    if trous:
        print(f"  ATTENTION : {len(trous)} trou(s) temporel(s) — donnees NON continues")
    print("=" * 62)
    return df


def run_collection(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    directory = segments_dir(out_path)
    directory.mkdir(exist_ok=True)

    total_seconds = args.hours * 3600 if args.hours else args.duration
    segment_seconds = args.segment_minutes * 60
    n_segments_total = max(1, int(-(-total_seconds // segment_seconds)))

    valides = [f for f in existing_segments(directory, args.symbol) if segment_valide(f)]
    invalides = [f for f in existing_segments(directory, args.symbol) if not segment_valide(f)]
    for f in invalides:
        logger.warning("segment corrompu supprime : %s", f.name)
        f.unlink()
    start_index = len(valides) + 1
    if valides and not args.resume and not args.force_restart:
        print(f"{len(valides)} segment(s) deja present(s) : ajoutez --resume pour continuer.")
        sys.exit(1)
    if args.force_restart:
        for f in valides:
            f.unlink()
        valides = []
        start_index = 1

    if args.resume and valides:
        print(f"REPRISE : {len(valides)} segment(s) valides, poursuite au segment {start_index}.")
    else:
        print(f"Collecte de {total_seconds/3600:.1f} h en {n_segments_total} segments de {segment_seconds:.0f}s.")

    consecutive_failures = 0
    for i in range(start_index, n_segments_total + 1):
        seg_path = directory / f"seg_{args.symbol}_{i:04d}.csv"
        print(f"[segment {i}/{n_segments_total}] collecte de {segment_seconds:.0f}s...")
        buffer = []
        try:
            ticks, buffer = collect_segment_with_retries(
                segment_seconds,
                args.symbol,
                args.interval,
                ticks_out=buffer,
            )
        except KeyboardInterrupt:
            if buffer:
                save_csv(buffer, str(seg_path))
                print(f"[segment {i}] interruption : {len(buffer)} ticks partiels sauvegardes.")
            print("Arret propre. Relancez avec --resume pour continuer.")
            sys.exit(130)
        if ticks:
            save_csv(ticks, str(seg_path))
            prices = [t["last_price"] for t in ticks]
            print(
                f"[segment {i}/{n_segments_total}] {len(ticks)} ticks | "
                f"prix {min(prices):.2f}-{max(prices):.2f} $"
            )
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"[segment {i}] ECHEC apres {MAX_SEGMENT_RETRIES} essais.")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(
                    f"{MAX_CONSECUTIVE_FAILURES} segments consecutifs vides : reseau down. "
                    "Arret propre — relancez avec --resume quand le reseau revient."
                )

    concatenate(directory, args.symbol, out_path)


def main():
    parser = argparse.ArgumentParser(description="Collecte longue segmentee avec reprise")
    parser.add_argument("--out", required=True, help="CSV final, ex: btc_24h.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hours", type=float, help="Duree totale en heures (ex: 24)")
    group.add_argument("--duration", type=int, help="Duree totale en secondes (ex: 86400)")
    parser.add_argument("--segment-minutes", type=float, default=7.5, help="Duree d'un segment en minutes (defaut 7.5)")
    parser.add_argument("--interval", type=float, default=0.25, help="Intervalle de polling en secondes")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--resume", action="store_true", help="Reprendre une collecte interrompue")
    parser.add_argument("--force-restart", action="store_true", help="Repartir de zero (supprime les segments existants)")
    parser.add_argument("--concat-only", action="store_true", help="Concatener les segments existants sans collecter")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    out_path = Path(args.out)
    if args.concat_only:
        concatenate(segments_dir(out_path), args.symbol, out_path)
        return
    run_collection(args)


if __name__ == "__main__":
    main()
