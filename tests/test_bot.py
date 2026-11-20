import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from bot import (
    BybitClient,
    Config,
    SignalStore,
    evaluate_price_alert,
    format_signal,
    get_scan_symbols,
)
from strategy import Signal


class ConfigTests(unittest.TestCase):
    def test_defaults_are_signal_only_and_configurable(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()

        self.assertTrue(config.all_usdt_perpetuals)
        self.assertEqual(config.symbols, ())
        self.assertEqual(config.interval, "15")
        self.assertEqual(config.scan_interval_seconds, 300)
        self.assertEqual((config.fast_ema_period, config.slow_ema_period), (20, 50))

    def test_explicit_symbols_override_all_markets(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "test-token", "SYMBOLS": "BTCUSDT,ETHUSDT,BTCUSDT"},
            clear=True,
        ):
            config = Config.from_env()

        self.assertFalse(config.all_usdt_perpetuals)
        self.assertEqual(config.symbols, ("BTCUSDT", "ETHUSDT"))

    def test_invalid_ema_order_is_rejected(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "test-token", "FAST_EMA_PERIOD": "50", "SLOW_EMA_PERIOD": "20"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "EMA periods"):
                Config.from_env()

    def test_signal_message_contains_entry_targets_and_stop(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        signal = Signal("BTCUSDT", "LONG", 100, 103, 106, 97, 0)

        message = format_signal(signal, config)
        self.assertIn("Вход: 100.00", message)
        self.assertIn("Тейк 1: 103.00", message)
        self.assertIn("Стоп: 97.0000", message)
        self.assertIn("бот не торгует", message)


class BybitClientTests(unittest.TestCase):
    def test_discovers_only_trading_usdt_perpetuals_across_pages(self):
        pages = {
            None: {
                "list": [
                    {"symbol": "BTCUSDT", "status": "Trading", "contractType": "LinearPerpetual", "settleCoin": "USDT"},
                    {"symbol": "BTCUSDC", "status": "Trading", "contractType": "LinearPerpetual", "settleCoin": "USDC"},
                    {"symbol": "PENDINGUSDT", "status": "PendingOpen", "contractType": "LinearPerpetual", "settleCoin": "USDT"},
                ],
                "nextPageCursor": "next",
            },
            "next": {
                "list": [
                    {"symbol": "ETHUSDT", "status": "Trading", "contractType": "LinearPerpetual", "settleCoin": "USDT"},
                    {"symbol": "BTCUSDT", "status": "Trading", "contractType": "LinearPerpetual", "settleCoin": "USDT"},
                    {"symbol": "FUTUREUSDT", "status": "Trading", "contractType": "LinearFutures", "settleCoin": "USDT"},
                ],
                "nextPageCursor": "",
            },
        }

        class FakeClient:
            async def get(self, url, params):
                cursor = params.get("cursor")
                result = pages[cursor]
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={"retCode": 0, "result": result},
                )

        async def discover():
            with patch("bot.BYBIT_REQUEST_INTERVAL_SECONDS", 0):
                return await BybitClient(FakeClient()).get_active_usdt_perpetual_symbols()

        self.assertEqual(asyncio.run(discover()), ("BTCUSDT", "ETHUSDT"))

    def test_missing_symbol_cache_is_refreshed_even_if_timestamp_is_recent(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()

        class FakeBybitClient:
            async def get_active_usdt_perpetual_symbols(self):
                return ("BTCUSDT", "ETHUSDT")

        context = SimpleNamespace(
            application=SimpleNamespace(
                bot_data={"symbols_refreshed_at": time.monotonic() - 10}
            )
        )

        symbols = asyncio.run(get_scan_symbols(context, config, FakeBybitClient()))

        self.assertEqual(symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(context.application.bot_data["active_symbols"], symbols)

    def test_discards_unclosed_candle_and_returns_chronological_data(self):
        rows = [
            [str(timestamp), "10", "12", "9", "11", "100", "1100"]
            for timestamp in (300, 200, 100)
        ]

        class FakeClient:
            async def get(self, url, params):
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={"retCode": 0, "result": {"list": rows}},
                )

        candles = asyncio.run(BybitClient(FakeClient()).get_closed_candles("BTCUSDT", "15"))
        self.assertEqual([candle.timestamp for candle in candles], [100, 200])

    def test_bybit_error_is_reported(self):
        class FakeClient:
            async def get(self, url, params):
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={"retCode": 10001, "retMsg": "invalid symbol", "result": {"list": []}},
                )

        with self.assertRaisesRegex(RuntimeError, "invalid symbol"):
            asyncio.run(BybitClient(FakeClient()).get_closed_candles("BADUSDT", "15"))

    def test_ticker_change_is_converted_to_percent(self):
        class FakeClient:
            async def get(self, url, params):
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={
                        "retCode": 0,
                        "result": {
                            "list": [
                                {"symbol": "BTCUSDT", "lastPrice": "65000", "price24hPcnt": "0.0525"},
                                {"symbol": "ETHUSDT", "lastPrice": "3000", "price24hPcnt": "-0.01"},
                            ]
                        },
                    },
                )

        tickers = asyncio.run(BybitClient(FakeClient()).get_tickers({"BTCUSDT"}))
        self.assertEqual(tickers, {"BTCUSDT": (65000.0, 5.25)})


class WatchlistTests(unittest.TestCase):
    def test_watchlist_is_chat_specific_and_can_be_removed(self):
        store = SignalStore(":memory:")
        store.subscribe(1)
        store.subscribe(2)
        store.watch(1, "BTCUSDT", 5)
        store.watch(2, "BTCUSDT", 3)

        self.assertEqual(store.watchlist(1), [("BTCUSDT", 5.0)])
        self.assertEqual(store.watchlist(2), [("BTCUSDT", 3.0)])
        self.assertEqual(
            store.watched_symbols(),
            [(1, "BTCUSDT", 5.0, None), (2, "BTCUSDT", 3.0, None)],
        )

        store.set_watch_alert_direction(1, "BTCUSDT", "up")
        self.assertEqual(store.watchlist(1), [("BTCUSDT", 5.0)])
        self.assertTrue(store.unwatch(1, "BTCUSDT"))
        self.assertEqual(store.watchlist(1), [])
        store.close()

    def test_price_alert_uses_threshold_and_hysteresis(self):
        self.assertEqual(evaluate_price_alert(5.1, 5, None), ("up", "up"))
        self.assertEqual(evaluate_price_alert(6.0, 5, "up"), (None, "up"))
        self.assertEqual(evaluate_price_alert(3.9, 5, "up"), (None, None))
        self.assertEqual(evaluate_price_alert(-5.2, 5, "up"), ("down", "down"))


if __name__ == "__main__":
    unittest.main()
