import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from bot import (
    BybitClient,
    Config,
    SignalStore,
    evaluate_price_alert,
    format_rsi_extreme,
    format_signal,
    get_market_symbols,
    handle_closed_candle,
    parse_confirmed_candle,
    subscription_batches,
)
from strategy import Candle, Signal


class ConfigTests(unittest.TestCase):
    def test_defaults_are_signal_only_and_configurable(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()

        self.assertTrue(config.all_usdt_perpetuals)
        self.assertEqual(config.symbols, ())
        self.assertEqual(config.interval, "15")
        self.assertEqual((config.fast_ema_period, config.slow_ema_period), (20, 50))
        self.assertEqual((config.trend_interval, config.trend_ema_period), ("60", 200))
        self.assertEqual((config.volume_multiplier, config.volume_lookback), (1.2, 20))

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
        self.assertIn("Время свечи: 1970-01-01 03:00 МСК", message)
        self.assertIn("бот не торгует", message)

    def test_overheat_alert_warns_without_promising_a_reversal(self):
        message = format_rsi_extreme("BTCUSDT", "overbought", 72.5, 65000, 0, "15")

        self.assertIn("ПЕРЕКУПЛЕННОСТЬ", message)
        self.assertIn("RSI 72.5", message)
        self.assertIn("Возможен откат вниз", message)
        self.assertIn("разворот не гарантирован", message)
        self.assertIn("Время свечи: 1970-01-01 03:00 МСК", message)


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

    def test_selects_top_active_perpetuals_by_turnover(self):
        class FakeClient:
            async def get(self, url, params):
                if url.endswith("instruments-info"):
                    response = {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {
                                    "symbol": symbol,
                                    "status": "Trading",
                                    "contractType": "LinearPerpetual",
                                    "settleCoin": "USDT",
                                }
                                for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
                            ],
                            "nextPageCursor": "",
                        },
                    }
                else:
                    response = {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {"symbol": "BTCUSDT", "turnover24h": "100"},
                                {"symbol": "ETHUSDT", "turnover24h": "200"},
                                {"symbol": "SOLUSDT", "turnover24h": "150"},
                                {"symbol": "NOTACTIVEUSDT", "turnover24h": "1000"},
                            ]
                        },
                    }
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json=response,
                )

        async def select():
            with patch("bot.BYBIT_REQUEST_INTERVAL_SECONDS", 0):
                return await BybitClient(FakeClient()).get_top_usdt_perpetual_symbols_by_turnover(
                    2
                )

        self.assertEqual(asyncio.run(select()), ("ETHUSDT", "SOLUSDT"))

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

        symbols = asyncio.run(
            get_market_symbols(context.application.bot_data, config, FakeBybitClient())
        )

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
        self.assertEqual(candles[0].volume, 100)

    def test_confirmed_websocket_candle_is_parsed_and_open_candle_is_ignored(self):
        message = {
            "topic": "kline.15.BTCUSDT",
            "data": [
                {
                    "start": "900000",
                    "open": "10",
                    "high": "12",
                    "low": "9",
                    "close": "11",
                    "volume": "25",
                    "confirm": True,
                }
            ],
        }

        parsed = parse_confirmed_candle(message)

        self.assertEqual(parsed, ("BTCUSDT", Candle(900000, 10, 12, 9, 11, 25)))
        message["data"][0]["confirm"] = False
        self.assertIsNone(parse_confirmed_candle(message))

    def test_websocket_topics_are_batched_under_bybit_argument_limit(self):
        topics = [f"kline.15.COIN{index}USDT" for index in range(1000)]

        batches = subscription_batches(topics)

        self.assertGreater(len(batches), 1)
        self.assertEqual([topic for batch in batches for topic in batch], topics)
        self.assertTrue(all(sum(len(topic) + 3 for topic in batch) <= 18000 for batch in batches))

    def test_duplicate_closed_websocket_candle_is_processed_once(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        store = SignalStore(":memory:")
        history = [Candle(index, 100, 101, 99, 100, 10) for index in range(60)]
        application = SimpleNamespace(
            bot_data={
                "config": config,
                "store": store,
                "candle_cache": {"15": {"BTCUSDT": history}},
            },
            bot=SimpleNamespace(),
        )
        candle = Candle(60, 100, 101, 99, 100, 10)

        async def process_twice():
            with patch("bot.build_signal", return_value=None) as build_signal_mock:
                await handle_closed_candle(application, "15", "BTCUSDT", candle)
                await handle_closed_candle(application, "15", "BTCUSDT", candle)
                return build_signal_mock.call_count

        self.assertEqual(asyncio.run(process_twice()), 1)
        self.assertEqual(store.last_processed("BTCUSDT", "15"), 60)
        self.assertTrue(application.bot_data["last_market_event_at"].endswith("МСК"))
        store.close()

    def test_historical_candles_are_fetched_with_window_and_volume(self):
        class FakeClient:
            async def get(self, url, params):
                self.params = params
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={
                        "retCode": 0,
                        "result": {
                            "list": [["100", "10", "12", "9", "11", "25", "275"]]
                        },
                    },
                )

        fake_client = FakeClient()
        candles = asyncio.run(
            BybitClient(fake_client).get_historical_candles(
                "BTCUSDT", "15", 50, 150, limit=10
            )
        )
        self.assertEqual(fake_client.params["start"], 50)
        self.assertEqual(fake_client.params["end"], 150)
        self.assertEqual(candles[0].volume, 25)

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

    def test_bybit_rate_limit_is_retried(self):
        class FakeClient:
            calls = 0

            async def get(self, url, params):
                self.calls += 1
                result = (
                    {"retCode": 10006, "retMsg": "Too many visits"}
                    if self.calls == 1
                    else {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {"symbol": "BTCUSDT", "lastPrice": "65000", "price24hPcnt": "0.01"}
                            ]
                        },
                    }
                )
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json=result,
                )

        fake_client = FakeClient()

        async def fetch_tickers():
            with (
                patch("bot.BYBIT_REQUEST_INTERVAL_SECONDS", 0),
                patch("bot.asyncio.sleep", new=AsyncMock()),
            ):
                return await BybitClient(fake_client).get_tickers({"BTCUSDT"})

        self.assertEqual(asyncio.run(fetch_tickers()), {"BTCUSDT": (65000.0, 1.0)})
        self.assertEqual(fake_client.calls, 2)

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

    def test_rsi_extreme_processing_state_is_persisted_per_symbol_and_interval(self):
        store = SignalStore(":memory:")

        self.assertIsNone(store.rsi_extreme_state("BTCUSDT", "15"))
        store.mark_rsi_extreme_processed("BTCUSDT", "15", 123, "overbought")
        self.assertEqual(
            store.rsi_extreme_state("BTCUSDT", "15"), (123, "overbought")
        )
        self.assertIsNone(store.rsi_extreme_state("ETHUSDT", "15"))
        self.assertIsNone(store.rsi_extreme_state("BTCUSDT", "60"))
        store.close()


if __name__ == "__main__":
    unittest.main()
