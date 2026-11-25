import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from bot import (
    SIGNAL_QUEUE_INTERVAL_SECONDS,
    BybitClient,
    Config,
    PendingMarketAlert,
    SignalStore,
    calculate_market_analysis,
    enqueue_market_alert,
    evaluate_price_alert,
    format_market_analysis,
    format_rsi_extreme,
    format_signal,
    get_market_symbols,
    handle_closed_candle,
    parse_confirmed_candle,
    broadcast_market_photo,
    render_market_chart,
    run_signal_dispatcher,
    subscription_batches,
)
from strategy import Candle, EMA_PULLBACK_STRATEGY_NAME, Signal


class ConfigTests(unittest.TestCase):
    def test_all_strategy_alert_cooldown_is_fixed_to_three_minutes(self):
        self.assertEqual(SIGNAL_QUEUE_INTERVAL_SECONDS, 180)

    def test_defaults_are_signal_only_and_configurable(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()

        self.assertTrue(config.all_usdt_perpetuals)
        self.assertEqual(config.symbols, ())
        self.assertEqual(config.interval, "15")
        self.assertEqual(config.ema_pullback_interval, "240")
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

    def test_pullback_interval_can_be_set_to_one_hour_or_four_hours_only(self):
        for interval in ("60", "240"):
            with patch.dict(
                os.environ,
                {"TELEGRAM_BOT_TOKEN": "test-token", "EMA_PULLBACK_INTERVAL": interval},
                clear=True,
            ):
                self.assertEqual(Config.from_env().ema_pullback_interval, interval)
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "test-token", "EMA_PULLBACK_INTERVAL": "15"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "EMA_PULLBACK_INTERVAL"):
                Config.from_env()

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

    def test_signal_message_adds_market_analysis(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        candles = [
            Candle(index * 900_000, 100 + index, 102 + index, 99 + index, 100 + index, 10)
            for index in range(100)
        ]
        signal = Signal("BTCUSDT", "LONG", 199, 202, 205, 196, candles[-1].timestamp)

        message = format_signal(signal, config, candles)

        self.assertIn("Анализ закрытых свечей", message)
        self.assertIn("изменение за 24ч", message)
        self.assertIn("RSI", message)
        self.assertIn("MA50", message)

    def test_pullback_signal_message_is_tagged_and_uses_its_own_interval(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        candles = [
            Candle(index * 14_400_000, 100, 102, 99, 101, 10)
            for index in range(100)
        ]
        signal = Signal(
            "BTCUSDT", "LONG", 101, 104, 107, 98, candles[-1].timestamp,
            strategy=EMA_PULLBACK_STRATEGY_NAME,
        )

        message = format_signal(signal, config, candles, interval="240")

        self.assertIn(f"Стратегия: {EMA_PULLBACK_STRATEGY_NAME}", message)
        self.assertIn("ОТДЕЛЬНАЯ КОНФЛЮЕНС-СТРАТЕГИЯ", message)
        self.assertIn("4ч", message)

    def test_market_analysis_reports_change_rsi_and_distance_from_ma50(self):
        candles = [
            Candle(index, 100 + index, 102 + index, 99 + index, 100 + index, 10)
            for index in range(100)
        ]

        analysis = calculate_market_analysis(candles, "15", 14)
        summary = format_market_analysis(analysis)

        self.assertAlmostEqual(analysis["current_price"], 199)
        self.assertAlmostEqual(analysis["change_24h_pct"], 96 / 103 * 100)
        self.assertIsNotNone(analysis["rsi"])
        self.assertIsNotNone(analysis["ma50_difference_pct"])
        self.assertIn("отклонение от MA50", summary)

    def test_market_chart_is_png_and_contains_signal_levels(self):
        candles = [
            Candle(
                index * 900_000,
                100 + index,
                102 + index,
                99 + index,
                101 + index,
                10,
            )
            for index in range(100)
        ]
        signal = Signal("BTCUSDT", "LONG", 200, 210, 220, 190, candles[-1].timestamp)

        image = render_market_chart(candles, "BTCUSDT", "15", signal)

        self.assertTrue(image.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_market_chart_is_sent_as_a_telegram_photo(self):
        candles = [
            Candle(index * 900_000, 100 + index, 102 + index, 99 + index, 101 + index, 10)
            for index in range(60)
        ]
        send_photo = AsyncMock()
        application = SimpleNamespace(bot=SimpleNamespace(send_photo=send_photo))

        asyncio.run(
            broadcast_market_photo(
                application,
                [123],
                "market analysis",
                candles,
                "BTCUSDT",
                "15",
            )
        )

        self.assertEqual(send_photo.await_count, 1)
        self.assertEqual(send_photo.await_args.kwargs["chat_id"], 123)
        self.assertEqual(
            send_photo.await_args.kwargs["photo"].filename,
            "btcusdt-analysis.png",
        )

    def test_overheat_alert_warns_without_promising_a_reversal(self):
        analysis = {
            "current_price": 65000,
            "change_24h_pct": 2.5,
            "change_period_hours": 24,
            "rsi": 72.5,
            "ma50": 64000,
            "ma50_difference_pct": 1.56,
        }
        message = format_rsi_extreme(
            "BTCUSDT", "overbought", 72.5, 65000, 0, "15", analysis
        )

        self.assertIn("ПЕРЕКУПЛЕННОСТЬ", message)
        self.assertIn("RSI 72.5", message)
        self.assertIn("Возможен откат вниз", message)
        self.assertIn("разворот не гарантирован", message)
        self.assertIn("Время свечи: 1970-01-01 03:00 МСК", message)
        self.assertIn("отклонение от MA50", message)


class SignalQueueTests(unittest.TestCase):
    def test_queue_keeps_distinct_strategies_for_same_symbol_and_candle(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        application = SimpleNamespace(
            bot_data={
                "config": config,
                "pending_alerts": [],
                "queued_alert_keys": set(),
            }
        )
        candles = [
            Candle(index, 100, 101, 99, 100, 10)
            for index in range(config.volume_lookback + 1)
        ]
        crossover_signal = Signal("BTCUSDT", "LONG", 100, 103, 106, 97, 20)
        pullback_signal = Signal(
            "BTCUSDT", "LONG", 100, 103, 106, 97, 20,
            strategy=EMA_PULLBACK_STRATEGY_NAME,
        )

        async def enqueue_both():
            application.bot_data["signal_queue_event"] = asyncio.Event()
            enqueue_market_alert(
                application, [123], "crossover", candles, "BTCUSDT", "240",
                crossover_signal,
            )
            enqueue_market_alert(
                application, [123], "pullback", candles, "BTCUSDT", "240",
                pullback_signal,
            )

        asyncio.run(enqueue_both())

        self.assertEqual(len(application.bot_data["pending_alerts"]), 2)
        self.assertEqual(len(application.bot_data["queued_alert_keys"]), 2)

    def test_dispatcher_skips_alerts_after_unsubscribe(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        store = SignalStore(":memory:")
        store.subscribe(123)
        application = SimpleNamespace(
            bot_data={
                "config": config,
                "store": store,
                "pending_alerts": [],
                "queued_alert_keys": set(),
            }
        )
        candles = [
            Candle(index, 100, 101, 99, 100 + index * 0.01, 10)
            for index in range(config.volume_lookback + 1)
        ]
        signal = Signal("BTCUSDT", "LONG", 100, 103, 106, 97, 20)

        async def exercise_queue():
            application.bot_data["signal_queue_event"] = asyncio.Event()
            with patch("bot.SIGNAL_QUEUE_GATHER_SECONDS", 0), patch(
                "bot.broadcast_market_photo", new=AsyncMock()
            ) as send:
                enqueue_market_alert(
                    application, [123], "queued", candles, "BTCUSDT", "15", signal
                )
                store.unsubscribe(123)
                task = asyncio.create_task(run_signal_dispatcher(application))
                try:
                    await asyncio.sleep(0.01)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                self.assertEqual(send.await_count, 0)

        asyncio.run(exercise_queue())
        self.assertEqual(len(application.bot_data["pending_alerts"]), 0)
        store.close()

    def test_dispatcher_prioritizes_signals_and_enforces_cooldown(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        store = SignalStore(":memory:")
        store.subscribe(123)
        candles = [
            Candle(index, 100, 101, 99, 100 + index * 0.01, 10)
            for index in range(config.volume_lookback + 1)
        ]
        low_priority_candles = candles[:-1] + [Candle(20, 100, 101, 99, 100, 12)]
        high_priority_candles = candles[:-1] + [Candle(20, 100, 101, 99, 100, 30)]
        application = SimpleNamespace(
            bot_data={
                "config": config,
                "store": store,
                "pending_alerts": [],
                "queued_alert_keys": set(),
            }
        )
        low_priority_signal = Signal("BTCUSDT", "LONG", 100, 103, 106, 97, 20)
        high_priority_signal = Signal("ETHUSDT", "LONG", 100, 103, 106, 97, 20)
        delivered = []
        stale_key = ("signal", "OLDUSDT", "15", 1, "EMA 20/50 crossover")

        async def record_delivery(*args):
            delivered.append((args[4], asyncio.get_running_loop().time()))

        async def exercise_queue():
            application.bot_data["signal_queue_event"] = asyncio.Event()
            with patch("bot.SIGNAL_QUEUE_GATHER_SECONDS", 0.02), patch(
                "bot.SIGNAL_QUEUE_INTERVAL_SECONDS", 0.03
            ), patch("bot.broadcast_market_photo", new=record_delivery):
                enqueue_market_alert(
                    application, [123], "low", low_priority_candles,
                    "BTCUSDT", "15", low_priority_signal,
                )
                enqueue_market_alert(
                    application, [123], "duplicate", low_priority_candles,
                    "BTCUSDT", "15", low_priority_signal,
                )
                enqueue_market_alert(
                    application, [123], "high", high_priority_candles,
                    "ETHUSDT", "15", high_priority_signal,
                )
                application.bot_data["queued_alert_keys"].add(stale_key)
                application.bot_data["pending_alerts"].append(
                    PendingMarketAlert(
                        (123,), "stale", tuple(candles), "OLDUSDT", "15",
                        low_priority_signal, 1, 0, 0, stale_key,
                    )
                )
                task = asyncio.create_task(run_signal_dispatcher(application))
                try:
                    await asyncio.sleep(0.09)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        asyncio.run(exercise_queue())

        self.assertEqual([symbol for symbol, _ in delivered], ["ETHUSDT", "BTCUSDT"])
        self.assertGreaterEqual(delivered[1][1] - delivered[0][1], 0.025)
        self.assertNotIn(stale_key, application.bot_data["queued_alert_keys"])
        self.assertEqual(len(application.bot_data["pending_alerts"]), 0)
        store.close()


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

    def test_pullback_strategy_processes_each_closed_candle_once(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True):
            config = Config.from_env()
        store = SignalStore(":memory:")
        store.subscribe(123)
        history = [
            Candle(index * 14_400_000, 100, 101, 99, 100, 10)
            for index in range(201)
        ]
        application = SimpleNamespace(
            bot_data={
                "config": config,
                "store": store,
                "candle_cache": {"240": {"BTCUSDT": history}},
            },
            bot=SimpleNamespace(),
        )
        candle = Candle(201 * 14_400_000, 100, 102, 99, 101, 12)
        signal = Signal(
            "BTCUSDT", "LONG", 101, 104, 107, 98, candle.timestamp,
            strategy=EMA_PULLBACK_STRATEGY_NAME,
        )

        async def process_twice():
            with (
                patch("bot.build_ema_pullback_signal", return_value=signal) as build_mock,
                patch("bot.enqueue_market_alert") as enqueue_mock,
            ):
                await handle_closed_candle(application, "240", "BTCUSDT", candle)
                await handle_closed_candle(application, "240", "BTCUSDT", candle)
                return build_mock, enqueue_mock

        build_mock, enqueue_mock = asyncio.run(process_twice())

        self.assertEqual(build_mock.call_count, 1)
        self.assertEqual(enqueue_mock.call_count, 1)
        self.assertEqual(enqueue_mock.call_args.args[5], "240")
        self.assertIn(EMA_PULLBACK_STRATEGY_NAME, enqueue_mock.call_args.args[2])
        self.assertEqual(
            store.last_strategy_processed(
                EMA_PULLBACK_STRATEGY_NAME, "BTCUSDT", "240"
            ),
            candle.timestamp,
        )
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
