"""Collecte longue de donnees BTC : segmentation, reprise et concatenation.

Problemes regles par rapport a binance_collector.py en direct :
- un processus de plusieurs heures est fragile (timeout terminal, veille Mac,
  crash) -> la collecte est decoupee en segments sauvegardes immediatement ;
- apres une interruption, --resume repart la ou la collecte s'etait arretee ;
- les segments sont concatenes automatiquement dans le CSV final.

Exemples :
    # 24 heures de collecte, segments de 7.5 min (defaut, sur)
    python3 collect_long.py --hours 24 --out btc_24h.csv

    # 8 heures avec segments de 10 min et reprise apres interruption
    python3 collect_long.py --hours 8 --out btc_8h.csv --segment-minutes 10 --resume

    # En arriere-plan sur Mac (immune a la fermeture du terminal)
    caffeinate -is nohup python3 collect_long.py --hours 24 --out btc_24h.csv \
        > collecte.log 2>&1 &
    tail -f collecte.log
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from binance_collector import collect, save_csv

DEFAULT_SEGMENT_SECONDS = 450


def segments_dir(out_path: Path) -> Path:
    return out_path.parent / (out_path.stem + "_segments")


def existing_segments(directory: Path, symbol: str) -> list:
    if not directory.exists():
        return []
    files = sorted(
        directory.glob(f"seg_{symbol}_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return files


def run_collection(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    directory = segments_dir(out_path)
    directory.mkdir(exist_ok=True)

    total_seconds = args.hours * 3600 if args.hours else args.duration
    segment_seconds = args.segment_minutes * 60 if args.segment_minutes else args.duration_segment or DEFAULT_SEGMENT_SECONDS
    n_segments_total = max(1, -(-total_seconds // segment_seconds))

    done = existing_segments(directory, args.symbol)
    start_index = len(done) + 1
    if done and not args.resume:
        print(f"{len(done)} segment(s) deja present(s) --resume pour continuer, ou changez --out.")
        if not args.force_restart:
            sys.exit(1)

    if args.resume and done:
        ticks_deja = sum(len(pd.read_csv(f)) for f in done)
        print(f"REPRISE : {len(done)} segment(s) deja collectes (~{ticks_deja} ticks), poursuite au segment {start_index}.")
    else:
        print(f"Collecte de {total_seconds/3600:.1f} h en {n_segments_total} segments de {segment_seconds}s.")

    collected_seconds = 0
    for i in range(start_index, int(n_segments_total) + 1):
        seg_path = directory / f"seg_{args.symbol}_{i:04d}.csv"
        print(f"[segment {i}/{int(n_segments_total)}] collecte de {segment_seconds}s...")
        segment_start = time.time()
        ticks = collect(
            segment_seconds,
            symbol=args.symbol,
            poll_interval=args.interval,
            verbose=False,
        )
        if ticks:
            save_csv(ticks, str(seg_path))
            collected_seconds += time.time() - segment_start
            prices = [t["last_price"] for t in ticks]
            print(
                f"[segment {i}/{int(n_segments_total)}] {len(ticks)} ticks | "
                f"prix {min(prices):.2f}-{max(prices):.2f} $"
            )
        else:
            print(f"[segment {i}] aucun tick collecte (verifiez la connexion), nouvel essai au prochain segment.")

    concatenate(directory, args.symbol, out_path)
    print(f"Termine : {collected_seconds/60:.1f} min de collecte effective.")


def concatenate(directory: Path, symbol: str, out_path: Path) -> pd.DataFrame:
    files = existing_segments(directory, symbol)
    if not files:
        raise SystemExit("Aucun segment collecte.")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_csv(out_path, index=False)
    duration_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    print("=" * 62)
    print(f"CSV final : {out_path}")
    print(f"  {len(df)} ticks | {duration_min:.1f} min couvertes | {len(files)} segments")
    print(f"  prix {df['last_price'].min():.2f} - {df['last_price'].max():.2f} $")
    print("=" * 62)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collecte longue segmentee avec reprise")
    parser.add_argument("--out", required=True, help="CSV final, ex: btc_24h.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hours", type=float, help="Duree totale en heures (ex: 24)")
    group.add_argument("--duration", type=int, help="Duree totale en secondes (ex: 86400)")
    parser.add_argument("--segment-minutes", type=float, default=None, help="Duree d'un segment en minutes (defaut 7.5)")
    parser.add_argument("--duration-segment", type=int, default=None, help="Duree d'un segment en secondes")
    parser.add_argument("--interval", type=float, default=0.25, help="Intervalle de polling en secondes")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--resume", action="store_true", help="Reprendre une collecte interrompue")
    parser.add_argument("--force-restart", action="store_true", help="Repartir de zero meme si des segments existent")
    parser.add_argument("--concat-only", action="store_true", help="Concatener les segments existants sans collecter")
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.concat_only:
        concatenate(segments_dir(out_path), args.symbol, out_path)
        return
    run_collection(args)


if __name__ == "__main__":
    main()
