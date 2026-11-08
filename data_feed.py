"""Connecteurs de donnees de marche : replay CSV, CCXT sandbox, stubs Tardis/CME.

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


CME_TICK_SIZES = {"NQ": 0.25, "ES": 0.25}


class CMECSVFeed:
    """Ticks CME (NQ/ES) au format CSV FirstRateData : DateTime,Price,Volume.

    FirstRateData fournit les trades centralises mais pas la profondeur L2.
    Les niveaux de carnet sont donc synthetises : best bid/ask autour du prix
    de trade (taille du tick du contrat) et tailles decroissantes ponderees par
    le volume du trade. Approximation documentee : pour de l'order flow reel
    en profondeur sur CME, utiliser IBKRFeed ou un flux CME MDP3.
    """

    def __init__(self, path, contract="NQ", depth=5, level_decay=0.7):
        self.path = path
        self.contract = contract
        self.depth = depth
        self.level_decay = level_decay

    def __iter__(self):
        df = pd.read_csv(self.path)
        cols = {c.lower().strip(): c for c in df.columns}
        dt_col = next((cols[c] for c in ("datetime", "date", "timestamp") if c in cols), None)
        px_col = next((cols[c] for c in ("price", "last", "close") if c in cols), None)
        vol_col = next((cols[c] for c in ("volume", "vol") if c in cols), None)
        if dt_col is None or px_col is None:
            raise ValueError(
                f"Colonnes attendues DateTime,Price[,Volume] — trouve : {list(df.columns)}"
            )
        tick_size = CME_TICK_SIZES.get(self.contract, 0.25)
        for _, row in df.iterrows():
            price = float(row[px_col])
            volume = float(row[vol_col]) if vol_col is not None else 1.0
            tick = {
                "timestamp": pd.to_datetime(row[dt_col], utc=True),
                "last_price": price,
                "last_volume": volume,
                "best_bid": price - tick_size,
                "best_ask": price + tick_size,
            }
            for i in range(1, self.depth + 1):
                size = max(1.0, volume * (self.level_decay ** (i - 1)))
                tick[f"bid_size_{i}"] = size
                tick[f"ask_size_{i}"] = size
            yield tick


class IBKRFeed:
    """Flux CME via Interactive Brokers (TWS / IB Gateway).

    Necessite `pip install ib_insync` et une connexion TWS/Gateway active
    avec l'API active (File > Global Configuration > API > Settings).
    Le module est importe paresseusement : le reste du pipeline fonctionne
    sans cette dependance.
    """

    def __init__(
        self,
        symbol="NQ",
        exchange="CME",
        currency="USD",
        host="127.0.0.1",
        port=7497,
        client_id=17,
        poll_interval=0.2,
        depth=5,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.host = host
        self.port = port
        self.client_id = client_id
        self.poll_interval = poll_interval
        self.depth = depth

    def _connect(self):
        try:
            from ib_insync import IB, Contract, MarketDatum
        except ImportError as exc:
            raise ImportError(
                "IBKRFeed necessite le package ib_insync : pip install ib_insync"
            ) from exc
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = "FUT"
        contract.exchange = self.exchange
        contract.currency = self.currency
        ib.qualifyContracts(contract)
        return ib, contract

    def __iter__(self):
        ib, contract = self._connect()
        ticker = ib.reqMktDepth(contract, numRows=self.depth)
        try:
            while True:
                ib.sleep(self.poll_interval)
                dom = ticker.dom
                if dom is None or (dom.bids and dom.asks):
                    bid = dom.bids[0] if dom and dom.bids else None
                    ask = dom.asks[0] if dom and dom.asks else None
                    tick = {
                        "timestamp": pd.Timestamp.now(tz="UTC"),
                        "last_price": float(ticker.last or (bid.price + ask.price) / 2 if bid and ask else 0.0),
                        "last_volume": float(getattr(ticker, "lastSize", 0) or 0.0),
                        "best_bid": float(bid.price) if bid else float("nan"),
                        "best_ask": float(ask.price) if ask else float("nan"),
                    }
                    for i in range(1, self.depth + 1):
                        b = dom.bids[i - 1] if dom and len(dom.bids) >= i else None
                        a = dom.asks[i - 1] if dom and len(dom.asks) >= i else None
                        tick[f"bid_size_{i}"] = float(b.size) if b else 0.0
                        tick[f"ask_size_{i}"] = float(a.size) if a else 0.0
                    yield tick
        finally:
            ib.disconnect()


class OandaXAUFeed:
    """Flux XAU/USD spot interbancaire via l'API REST OANDA (v20).

    Sans dependance externe (urllib). Necessite la variable d'environnement
    OANDA_API_KEY ; utilise l'environnement "practice" par defaut.
    Recupere la bougie la plus courte (S5) et la prix mid bid/ask.
    """

    def __init__(
        self,
        instrument="XAU_USD",
        granularity="S5",
        poll_interval=5.0,
        api_key_env="OANDA_API_KEY",
        practice=True,
    ):
        self.instrument = instrument
        self.granularity = granularity
        self.poll_interval = poll_interval
        self.api_key_env = api_key_env
        self.practice = practice

    def _fetch(self):
        import json
        import os
        import urllib.request

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Variable d'environnement {self.api_key_env} manquante pour OandaXAUFeed"
            )
        host = "api-fxpractice.oanda.com" if self.practice else "api-fxtrade.oanda.com"
        url = (
            f"https://{host}/v3/instruments/{self.instrument}/candles"
            f"?granularity={self.granularity}&count=1&price=M"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
        candle = payload["candles"][0]
        mid = candle["mid"]
        return {
            "timestamp": pd.to_datetime(candle["time"], utc=True),
            "last_price": float(mid["c"]),
            "last_volume": float(candle["volume"]),
            "best_bid": float(mid["c"]) - 0.15,
            "best_ask": float(mid["c"]) + 0.15,
            "bid_size_1": 10.0,
            "ask_size_1": 10.0,
        }

    def __iter__(self):
        while True:
            try:
                yield self._fetch()
            except Exception:
                raise
            time.sleep(self.poll_interval)
