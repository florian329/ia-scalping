"""Connecteurs de donnees de marche BTC : replay CSV, CCXT sandbox, Tardis CSV.

Interface commune : un feed est un iterable de ticks (dicts) avec les cles
last_price, last_volume, best_bid, best_ask et bid_size_i / ask_size_i.
"""

import time

import pandas as pd

REQUIRED_TICK_KEYS = ("last_price", "last_volume")


class CSVReplayFeed:
    """Rejoue un CSV de ticks (meme format que data_generator)."""

    def __init__(self, path, speed=0.0):
        self.path = path
        self.speed = speed

    def __iter__(self):
        df = pd.read_csv(self.path)
        for _, row in df.iterrows():
            tick = {k: row[k] for k in df.columns if pd.notna(row[k])}
            if self.speed > 0:
                time.sleep(self.speed)
            yield tick


class ListFeed:
    """Feed en memoire, pratique pour les tests et la simulation rapide."""

    def __init__(self, ticks):
        self.ticks = list(ticks)

    def __iter__(self):
        yield from self.ticks


class ReconnectingFeed:
    """Enveloppe un feed : si l'iteration leve une exception, reessaie apres
    un delai croissant (backoff), jusqu'a max_retries."""

    def __init__(self, feed_factory, max_retries=5, base_delay=1.0, max_delay=30.0):
        self.feed_factory = feed_factory
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.reconnect_count = 0

    def __iter__(self):
        retries = 0
        while retries <= self.max_retries:
            try:
                for tick in self.feed_factory():
                    retries = 0
                    yield tick
                return
            except StopIteration:
                return
            except Exception as exc:
                retries += 1
                self.reconnect_count += 1
                if retries > self.max_retries:
                    raise
                delay = min(self.base_delay * (2 ** (retries - 1)), self.max_delay)
                print(f"[feed] Erreur ({exc.__class__.__name__}: {exc}), reconnexion {retries}/{self.max_retries} dans {delay:.1f}s")
                time.sleep(delay)


class CCXTFeed:
    """Feed temps reel via CCXT (exchange sandbox/testnet pour le paper trading).

    Necessite `pip install ccxt`. Le module est importe paresseusement pour
    garder le reste du pipeline utilisable sans cette dependance.
    """

    def __init__(self, exchange_id="binance", symbol="BTC/USDT", sandbox=True, poll_interval=0.2, depth=5):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.sandbox = sandbox
        self.poll_interval = poll_interval
        self.depth = depth

    def _make_exchange(self):
        import ccxt

        exchange = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        if self.sandbox and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return exchange

    def _poll(self, exchange):
        orderbook = exchange.fetch_order_book(self.symbol, limit=self.depth)
        ticker = exchange.fetch_ticker(self.symbol)
        tick = {
            "timestamp": pd.to_datetime(orderbook["timestamp"], unit="ms", utc=True),
            "best_bid": float(orderbook["bids"][0][0]),
            "best_ask": float(orderbook["asks"][0][0]),
            "last_price": float(ticker["last"]),
            "last_volume": float(ticker.get("lastVolume") or 0.0),
        }
        for i, (bid, _) in enumerate(orderbook["bids"][: self.depth], start=1):
            tick[f"bid_size_{i}"] = float(bid[1])
        for i, (ask, _) in enumerate(orderbook["asks"][: self.depth], start=1):
            tick[f"ask_size_{i}"] = float(ask[1])
        return tick

    def __iter__(self):
        exchange = self._make_exchange()
        while True:
            try:
                yield self._poll(exchange)
            except Exception:
                exchange = self._make_exchange()
            time.sleep(self.poll_interval)


class TardisCSVFeed:
    """Ingestion des CSV Tardis.dev (incremental_book_L2 / trades) pour BTC.

    Format attendu (colonnes Tardis standard) pour le carnet :
    timestamp, price, amount, side (bid/ask) — agrege en profondeur par tick.
    Pour les trades : timestamp, price, amount.
    """

    def __init__(self, book_csv, trades_csv, depth=10):
        self.book_csv = book_csv
        self.trades_csv = trades_csv
        self.depth = depth

    def __iter__(self):
        trades = pd.read_csv(self.trades_csv)
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        book = pd.read_csv(self.book_csv)
        book["timestamp"] = pd.to_datetime(book["timestamp"], utc=True)
        book = book.sort_values("timestamp")

        for _, trade in trades.iterrows():
            tick = {
                "timestamp": trade["timestamp"],
                "last_price": float(trade["price"]),
                "last_volume": float(trade["amount"]),
            }
            snapshot = book[book["timestamp"] <= trade["timestamp"]].tail(self.depth * 2)
            bids = snapshot[snapshot["side"].str.lower() == "bid"].tail(self.depth)
            asks = snapshot[snapshot["side"].str.lower() == "ask"].tail(self.depth)
            for i, (_, row) in enumerate(bids.iterrows(), start=1):
                tick[f"bid_size_{i}"] = float(row["amount"])
                if i == 1:
                    tick["best_bid"] = float(row["price"])
            for i, (_, row) in enumerate(asks.iterrows(), start=1):
                tick[f"ask_size_{i}"] = float(row["amount"])
                if i == 1:
                    tick["best_ask"] = float(row["price"])
            yield tick

