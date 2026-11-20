from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import websockets
from websockets.exceptions import WebSocketException
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from strategy import Candle, Signal, build_signal, confirm_trend, detect_rsi_extreme

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info"
BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"
BYBIT_LINEAR_WS_URL = "wss://stream.bybit.com/v5/public/linear"
VALID_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
SYMBOL_CACHE_SECONDS = 60 * 60
MAX_CONCURRENT_SYMBOL_SCANS = 8
MAX_SUBSCRIPTION_ARGUMENT_CHARS = 18000
WS_SYMBOL_REFRESH_SECONDS = 60 * 60
BYBIT_REQUEST_INTERVAL_SECONDS = 0.12
BYBIT_HTTP_TIMEOUT_SECONDS = 30
PRICE_ALERT_SCAN_SECONDS = 60
DEFAULT_PRICE_ALERT_THRESHOLD = 5.0
MIN_PRICE_ALERT_THRESHOLD = 0.1
MAX_PRICE_ALERT_THRESHOLD = 1000.0
LOG = logging.getLogger("bybit_signals")
MOSCOW_TIMEZONE = timezone(timedelta(hours=3), name="MSK")


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    symbols: tuple[str, ...]
    all_usdt_perpetuals: bool
    interval: str
    database_path: str
    fast_ema_period: int
    slow_ema_period: int
    rsi_period: int
    atr_period: int
    atr_multiplier: float
    take_profit_1_r: float
    take_profit_2_r: float
    volume_multiplier: float
    volume_lookback: int
    trend_interval: str
    trend_ema_period: int

    @classmethod
    def from_env(cls) -> Config:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        symbol_setting = os.environ.get("SYMBOLS", "ALL").strip()
        all_usdt_perpetuals = symbol_setting.upper() == "ALL"
        symbols = () if all_usdt_perpetuals else tuple(
            dict.fromkeys(
                symbol.strip().upper()
                for symbol in symbol_setting.split(",")
                if symbol.strip()
            )
        )
        if not all_usdt_perpetuals and (
            not symbols or any(not symbol.endswith("USDT") for symbol in symbols)
        ):
            raise ValueError("SYMBOLS must be ALL or a comma-separated list of USDT symbols")

        interval = os.environ.get("INTERVAL", "15").strip().upper()
        if interval not in VALID_INTERVALS:
            raise ValueError(f"INTERVAL must be one of: {', '.join(sorted(VALID_INTERVALS))}")

        try:
            fast_ema = int(os.environ.get("FAST_EMA_PERIOD", "20"))
            slow_ema = int(os.environ.get("SLOW_EMA_PERIOD", "50"))
            rsi_period = int(os.environ.get("RSI_PERIOD", "14"))
            atr_period = int(os.environ.get("ATR_PERIOD", "14"))
            atr_multiplier = float(os.environ.get("ATR_MULTIPLIER", "1.5"))
            take_profit_1_r = float(os.environ.get("TAKE_PROFIT_1_R", "1"))
            take_profit_2_r = float(os.environ.get("TAKE_PROFIT_2_R", "2"))
            volume_multiplier = float(os.environ.get("VOLUME_MULTIPLIER", "1.2"))
            volume_lookback = int(os.environ.get("VOLUME_LOOKBACK", "20"))
            trend_interval = os.environ.get("TREND_INTERVAL", "60").strip().upper()
            trend_ema_period = int(os.environ.get("TREND_EMA_PERIOD", "200"))
        except ValueError as error:
            raise ValueError("Strategy periods and multipliers must be numeric") from error
        if fast_ema < 2 or slow_ema <= fast_ema or slow_ema > 198:
            raise ValueError(
                "EMA periods must satisfy 2 <= FAST_EMA_PERIOD < SLOW_EMA_PERIOD <= 198"
            )
        if rsi_period < 2 or atr_period < 1:
            raise ValueError("RSI_PERIOD must be at least 2 and ATR_PERIOD must be positive")
        if trend_interval not in VALID_INTERVALS:
            raise ValueError(f"TREND_INTERVAL must be one of: {', '.join(sorted(VALID_INTERVALS))}")
        if not 2 <= trend_ema_period <= 500:
            raise ValueError("TREND_EMA_PERIOD must be between 2 and 500")
        if volume_lookback < 1 or not math.isfinite(volume_multiplier) or volume_multiplier <= 0:
            raise ValueError("VOLUME_LOOKBACK and VOLUME_MULTIPLIER must be positive")
        if (
            not all(math.isfinite(value) for value in (atr_multiplier, take_profit_1_r, take_profit_2_r))
            or atr_multiplier <= 0
            or take_profit_1_r <= 0
            or take_profit_2_r <= take_profit_1_r
        ):
            raise ValueError("Risk multipliers must be positive and TAKE_PROFIT_2_R > TAKE_PROFIT_1_R")

        return cls(
            telegram_bot_token=token,
            symbols=symbols,
            all_usdt_perpetuals=all_usdt_perpetuals,
            interval=interval,
            database_path=os.environ.get("DATABASE_PATH", "signals.sqlite3"),
            fast_ema_period=fast_ema,
            slow_ema_period=slow_ema,
            rsi_period=rsi_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            take_profit_1_r=take_profit_1_r,
            take_profit_2_r=take_profit_2_r,
            volume_multiplier=volume_multiplier,
            volume_lookback=volume_lookback,
            trend_interval=trend_interval,
            trend_ema_period=trend_ema_period,
        )


class SignalStore:
    def __init__(self, path: str) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS subscribers (
                chat_id INTEGER PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS processed_candles (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval)
            );
            CREATE TABLE IF NOT EXISTS processed_rsi_extremes (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (symbol, interval)
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                chat_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                threshold REAL NOT NULL,
                last_alert_direction TEXT,
                PRIMARY KEY (chat_id, symbol)
            );
            """
        )
        self.connection.commit()

    def subscribe(self, chat_id: int) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO subscribers(chat_id) VALUES (?)", (chat_id,)
        )
        self.connection.commit()

    def unsubscribe(self, chat_id: int) -> None:
        self.connection.execute("DELETE FROM subscribers WHERE chat_id = ?", (chat_id,))
        self.connection.commit()

    def subscribers(self) -> list[int]:
        rows = self.connection.execute("SELECT chat_id FROM subscribers").fetchall()
        return [row[0] for row in rows]

    def last_processed(self, symbol: str, interval: str) -> int | None:
        row = self.connection.execute(
            "SELECT timestamp FROM processed_candles WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
        return row[0] if row else None

    def mark_processed(self, symbol: str, interval: str, timestamp: int) -> None:
        self.connection.execute(
            """
            INSERT INTO processed_candles(symbol, interval, timestamp) VALUES (?, ?, ?)
            ON CONFLICT(symbol, interval) DO UPDATE SET timestamp = excluded.timestamp
            """,
            (symbol, interval, timestamp),
        )
        self.connection.commit()

    def rsi_extreme_state(self, symbol: str, interval: str) -> tuple[int, str] | None:
        row = self.connection.execute(
            """
            SELECT timestamp, state FROM processed_rsi_extremes
            WHERE symbol = ? AND interval = ?
            """,
            (symbol, interval),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def mark_rsi_extreme_processed(
        self, symbol: str, interval: str, timestamp: int, state: str
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO processed_rsi_extremes(symbol, interval, timestamp, state)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, interval) DO UPDATE SET
                timestamp = excluded.timestamp,
                state = excluded.state
            """,
            (symbol, interval, timestamp, state),
        )
        self.connection.commit()

    def watch(self, chat_id: int, symbol: str, threshold: float) -> None:
        self.connection.execute(
            """
            INSERT INTO watchlist(chat_id, symbol, threshold, last_alert_direction)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT(chat_id, symbol) DO UPDATE SET
                threshold = excluded.threshold,
                last_alert_direction = NULL
            """,
            (chat_id, symbol, threshold),
        )
        self.connection.commit()

    def unwatch(self, chat_id: int, symbol: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM watchlist WHERE chat_id = ? AND symbol = ?", (chat_id, symbol)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def watchlist(self, chat_id: int) -> list[tuple[str, float]]:
        rows = self.connection.execute(
            "SELECT symbol, threshold FROM watchlist WHERE chat_id = ? ORDER BY symbol",
            (chat_id,),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def watched_symbols(self) -> list[tuple[int, str, float, str | None]]:
        rows = self.connection.execute(
            """
            SELECT watchlist.chat_id, watchlist.symbol, watchlist.threshold,
                   watchlist.last_alert_direction
            FROM watchlist
            JOIN subscribers ON subscribers.chat_id = watchlist.chat_id
            ORDER BY watchlist.chat_id, watchlist.symbol
            """
        ).fetchall()
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    def set_watch_alert_direction(
        self, chat_id: int, symbol: str, direction: str | None
    ) -> None:
        self.connection.execute(
            "UPDATE watchlist SET last_alert_direction = ? WHERE chat_id = ? AND symbol = ?",
            (direction, chat_id, symbol),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class BybitClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._request_lock: asyncio.Lock | None = None
        self._last_request_at = 0.0

    async def get_active_usdt_perpetual_symbols(self) -> tuple[str, ...]:
        symbols = set()
        cursor = None
        seen_cursors = set()

        while True:
            params = {"category": "linear", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = await self._get_payload(BYBIT_INSTRUMENTS_URL, params)
            result = payload.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("list"), list):
                raise RuntimeError("Bybit returned malformed instruments data")

            for instrument in result["list"]:
                if not isinstance(instrument, dict):
                    raise RuntimeError("Bybit returned malformed instrument data")
                symbol = instrument.get("symbol")
                if (
                    instrument.get("status") == "Trading"
                    and instrument.get("contractType") == "LinearPerpetual"
                    and instrument.get("settleCoin") == "USDT"
                    and isinstance(symbol, str)
                    and symbol.endswith("USDT")
                ):
                    symbols.add(symbol)

            next_cursor = result.get("nextPageCursor")
            if not next_cursor:
                break
            if not isinstance(next_cursor, str):
                raise RuntimeError("Bybit returned an invalid instruments cursor")
            if next_cursor in seen_cursors:
                raise RuntimeError("Bybit returned a repeated instruments cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if not symbols:
            raise RuntimeError("Bybit returned no active USDT perpetual contracts")
        return tuple(sorted(symbols))

    async def get_closed_candles(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[Candle]:
        payload = await self._get_payload(
            BYBIT_KLINE_URL,
            {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit},
        )
        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise RuntimeError(f"Bybit returned malformed kline data for {symbol}")
        rows = result["list"]
        if len(rows) < 3:
            return []

        # Bybit returns newest first; drop the still-forming candle and restore chronological order.
        closed_rows = reversed(rows[1:])
        try:
            return [
                Candle(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in closed_rows
            ]
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Bybit returned malformed kline data for {symbol}") from error

    async def get_historical_candles(
        self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000
    ) -> list[Candle]:
        payload = await self._get_payload(
            BYBIT_KLINE_URL,
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": end_ms,
                "limit": min(max(limit, 1), 1000),
            },
        )
        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise RuntimeError(f"Bybit returned malformed historical data for {symbol}")
        try:
            return [
                Candle(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in result["list"]
            ]
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Bybit returned malformed historical data for {symbol}") from error

    async def get_tickers(self, symbols: set[str]) -> dict[str, tuple[float, float]]:
        if not symbols:
            return {}
        payload = await self._get_payload(BYBIT_TICKERS_URL, {"category": "linear"})
        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise RuntimeError("Bybit returned malformed ticker data")

        tickers = {}
        for ticker in result["list"]:
            if not isinstance(ticker, dict):
                raise RuntimeError("Bybit returned malformed ticker data")
            symbol = ticker.get("symbol")
            if symbol not in symbols:
                continue
            try:
                last_price = float(ticker["lastPrice"])
                change_24h_pct = float(ticker["price24hPcnt"]) * 100
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Bybit returned malformed ticker data for {symbol}") from error
            if not math.isfinite(last_price) or not math.isfinite(change_24h_pct):
                raise RuntimeError(f"Bybit returned invalid ticker values for {symbol}")
            tickers[symbol] = (last_price, change_24h_pct)
        return tickers

    async def _get_payload(self, url: str, params: dict[str, str | int]) -> dict:
        if self._request_lock is None:
            self._request_lock = asyncio.Lock()
        last_error: Exception | None = None
        for attempt in range(3):
            async with self._request_lock:
                now = asyncio.get_running_loop().time()
                delay = BYBIT_REQUEST_INTERVAL_SECONDS - (now - self._last_request_at)
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last_request_at = asyncio.get_running_loop().time()

            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.TransportError as error:
                last_error = error
                if attempt == 2:
                    raise
                await asyncio.sleep(1 + attempt)
        else:
            raise RuntimeError("Bybit request failed") from last_error
        if not isinstance(payload, dict) or payload.get("retCode") != 0:
            code = payload.get("retCode") if isinstance(payload, dict) else "unknown"
            message = payload.get("retMsg") if isinstance(payload, dict) else "invalid response"
            raise RuntimeError(f"Bybit returned retCode={code}: {message}")
        return payload


def format_signal(signal: Signal, config: Config) -> str:
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
    opened_at = datetime.fromtimestamp(signal.timestamp / 1000, tz=MOSCOW_TIMEZONE)
    interval_label = {"D": "1д", "W": "1н", "M": "1м"}.get(config.interval, f"{config.interval}м")
    return (
        f"🚨 СИГНАЛ • {signal.direction} {direction_emoji}\n"
        f"#{signal.symbol} · Bybit Perpetual · {interval_label}\n"
        f"Время свечи: {opened_at:%Y-%m-%d %H:%M} МСК\n\n"
        f"💰 Вход: {format_price(signal.entry)}\n"
        f"🎯 Тейк 1: {format_price(signal.take_profit_1)} "
        f"({price_change(signal.entry, signal.take_profit_1, signal.direction):+.2f}%)\n"
        f"🎯 Тейк 2: {format_price(signal.take_profit_2)} "
        f"({price_change(signal.entry, signal.take_profit_2, signal.direction):+.2f}%)\n"
        f"🛑 Стоп: {format_price(signal.stop_loss)} "
        f"({price_change(signal.entry, signal.stop_loss, signal.direction):+.2f}%)\n\n"
        f"EMA {config.fast_ema_period}/{config.slow_ema_period} + RSI {config.rsi_period}; "
        f"подтверждены EMA {config.trend_ema_period} на {config.trend_interval} и объемом "
        f"≥{config.volume_multiplier:g}x среднего. Уровни рассчитаны по ATR {config.atr_period}.\n"
        "Только аналитическая информация: бот не торгует и не является финансовой рекомендацией."
    )


def format_rsi_extreme(
    symbol: str,
    state: str,
    rsi_value: float,
    close: float,
    timestamp: int,
    interval: str,
) -> str:
    opened_at = datetime.fromtimestamp(timestamp / 1000, tz=MOSCOW_TIMEZONE)
    interval_label = {"D": "1д", "W": "1н", "M": "1м"}.get(interval, f"{interval}м")
    if state == "overbought":
        heading = "ПЕРЕКУПЛЕННОСТЬ"
        note = "Возможен откат вниз, но разворот не гарантирован."
        emoji = "🔥"
    else:
        heading = "ПЕРЕПРОДАННОСТЬ"
        note = "Возможен отскок вверх, но разворот не гарантирован."
        emoji = "🧊"
    return (
        f"{emoji} RSI-СИГНАЛ • {heading}\n"
        f"#{symbol} · Bybit Perpetual · {interval_label}\n"
        f"RSI {rsi_value:.1f} · Цена закрытия свечи: {format_price(close)} USDT\n"
        f"Время свечи: {opened_at:%Y-%m-%d %H:%M} МСК\n\n"
        f"{note}\n"
        "Это индикаторное предупреждение, а не рекомендация открывать сделку."
    )


def format_price(price: float) -> str:
    return f"{price:.2f}" if price >= 100 else f"{price:.4f}" if price >= 1 else f"{price:.8f}".rstrip("0")


def price_change(entry: float, level: float, direction: str) -> float:
    change = (level - entry) / entry * 100
    return change if direction == "LONG" else -change


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    store: SignalStore = context.application.bot_data["store"]
    store.subscribe(update.effective_chat.id)
    await update.message.reply_text(
        "Подписка включена. Я отслеживаю публичные свечи Bybit и отправляю технические сигналы.\n"
        "Команды: /status, /symbols, /watch, /watchlist, /unwatch, /stop, /help."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    store: SignalStore = context.application.bot_data["store"]
    store.unsubscribe(update.effective_chat.id)
    await update.message.reply_text("Подписка отключена. Для возобновления отправьте /start.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Я отслеживаю публичные USDT perpetual-контракты Bybit и сообщаю о пересечениях EMA "
            "с подтверждением RSI. Также сообщаю о входе RSI в зоны перекупленности (≥70) "
            "и перепроданности (≤30); это предупреждение о возможном откате/отскоке, "
            "не гарантия разворота. Сигналы на вход дополнительно фильтруются EMA тренда "
            "и повышенным объемом.\n"
            "Я не принимаю ордера и не использую ключи биржи.\n\n"
            "/start — подписаться\n/status — состояние бота\n/symbols — список рынков\n"
            "/watch BTCUSDT [5] — следить за монетой и задать порог изменения за 24 часа в %\n"
            "/watchlist — мой список отслеживания\n/unwatch BTCUSDT — удалить монету из списка\n"
            "/stop — отписаться от всех сообщений"
        )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    config: Config = context.application.bot_data["config"]
    last_event = context.application.bot_data.get(
        "last_market_event_at", "закрытых свечей пока не получено"
    )
    stream_status = context.application.bot_data.get("market_stream_status", "initializing")
    active_symbols = context.application.bot_data.get("active_symbols", ())
    markets = (
        f"Все активные USDT perpetual Bybit ({len(active_symbols)} рынков)"
        if config.all_usdt_perpetuals and active_symbols
        else "Все активные USDT perpetual Bybit (список загружается)"
        if config.all_usdt_perpetuals
        else f"{len(config.symbols)} рынков: {', '.join(config.symbols)}"
    )
    await update.message.reply_text(
        f"Режим: сигналы без торговли\n"
        f"Рынки: {markets}\n"
        f"Таймфрейм: {config.interval}\n"
        f"Стратегия: EMA {config.fast_ema_period}/{config.slow_ema_period}, "
        f"RSI {config.rsi_period} (перегрев ≥70 / ≤30), "
        f"ATR {config.atr_period} x{config.atr_multiplier:g}\n"
        f"Фильтры: EMA {config.trend_ema_period} ({config.trend_interval}), "
        f"объем ≥ {config.volume_multiplier:g}x среднего за {config.volume_lookback} свечей\n"
        f"Тейки: {config.take_profit_1_r:g}R / {config.take_profit_2_r:g}R\n"
        f"Поток Bybit: {stream_status}\n"
        f"Последняя закрытая свеча: {last_event}"
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    config: Config = context.application.bot_data["config"]
    active_symbols = context.application.bot_data.get("active_symbols", ())
    if config.all_usdt_perpetuals:
        if not active_symbols:
            await update.message.reply_text(
                "Список активных USDT perpetual-контрактов Bybit загружается при запуске потока."
            )
            return
        symbols_preview = ", ".join(active_symbols[:30])
        extra_count = len(active_symbols) - min(30, len(active_symbols))
        suffix = f" и еще {extra_count}" if extra_count else ""
        await update.message.reply_text(
            f"Отслеживаю {len(active_symbols)} активных USDT perpetual-контрактов.\n"
            f"Примеры: {symbols_preview}{suffix}\nТаймфрейм: {config.interval}"
        )
        return
    await update.message.reply_text(
        f"Отслеживаемые рынки: {', '.join(config.symbols)}\nТаймфрейм: {config.interval}"
    )


async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    if not context.args or len(context.args) > 2:
        await update.message.reply_text(
            f"Формат: /watch BTCUSDT [порог_в_%]. По умолчанию порог "
            f"{DEFAULT_PRICE_ALERT_THRESHOLD:g}% за 24 часа."
        )
        return

    symbol = context.args[0].strip().upper()
    if not symbol.endswith("USDT") or not symbol[:-4].isalnum():
        await update.message.reply_text("Укажите символ USDT perpetual, например /watch BTCUSDT 5")
        return
    try:
        threshold = (
            float(context.args[1])
            if len(context.args) == 2
            else DEFAULT_PRICE_ALERT_THRESHOLD
        )
    except ValueError:
        await update.message.reply_text("Порог должен быть числом, например /watch BTCUSDT 5")
        return
    if (
        not math.isfinite(threshold)
        or not MIN_PRICE_ALERT_THRESHOLD <= threshold <= MAX_PRICE_ALERT_THRESHOLD
    ):
        await update.message.reply_text(
            f"Порог должен быть от {MIN_PRICE_ALERT_THRESHOLD:g}% до "
            f"{MAX_PRICE_ALERT_THRESHOLD:g}%."
        )
        return

    client: BybitClient = context.application.bot_data["bybit"]
    try:
        ticker = await client.get_tickers({symbol})
    except (httpx.HTTPError, RuntimeError) as error:
        LOG.warning("Could not validate watch symbol %s: %s", symbol, error)
        await update.message.reply_text("Не удалось проверить Bybit. Попробуйте позже.")
        return
    if symbol not in ticker:
        await update.message.reply_text(
            f"Не нашел активный USDT perpetual-контракт {symbol} на Bybit."
        )
        return

    store: SignalStore = context.application.bot_data["store"]
    store.subscribe(update.effective_chat.id)
    store.watch(update.effective_chat.id, symbol, threshold)
    await update.message.reply_text(
        f"Добавил {symbol} в ваш список. Уведомлю, когда изменение за 24 часа "
        f"достигнет +{threshold:g}% или -{threshold:g}%.\n"
        f"Текущая цена: {format_price(ticker[symbol][0])} USDT."
    )


async def unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    if len(context.args) != 1:
        await update.message.reply_text("Формат: /unwatch BTCUSDT")
        return
    symbol = context.args[0].strip().upper()
    store: SignalStore = context.application.bot_data["store"]
    if store.unwatch(update.effective_chat.id, symbol):
        await update.message.reply_text(f"{symbol} удалена из вашего списка.")
    else:
        await update.message.reply_text(f"{symbol} нет в вашем списке отслеживания.")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    store: SignalStore = context.application.bot_data["store"]
    entries = store.watchlist(update.effective_chat.id)
    if not entries:
        await update.message.reply_text(
            "Ваш список пуст. Добавьте монету командой /watch BTCUSDT [порог_в_%]."
        )
        return
    lines = [
        f"{symbol} — уведомление при изменении за 24 ч. ±{threshold:g}%"
        for symbol, threshold in entries
    ]
    await update.message.reply_text("Ваш список отслеживания:\n" + "\n".join(lines))


def evaluate_price_alert(
    change_24h_pct: float, threshold: float, last_direction: str | None
) -> tuple[str | None, str | None]:
    if last_direction and abs(change_24h_pct) < threshold * 0.8:
        last_direction = None

    direction = (
        "up" if change_24h_pct >= threshold
        else "down" if change_24h_pct <= -threshold
        else None
    )
    if direction == last_direction:
        return None, last_direction
    if direction is not None:
        return direction, direction
    return None, last_direction


async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE) -> None:
    store: SignalStore = context.application.bot_data["store"]
    client: BybitClient = context.application.bot_data["bybit"]
    entries = store.watched_symbols()
    if not entries:
        return

    symbols_to_check = {symbol for _, symbol, _, _ in entries}
    try:
        tickers = await client.get_tickers(symbols_to_check)
    except (httpx.HTTPError, RuntimeError):
        LOG.exception("Failed to fetch ticker prices for watchlists")
        return

    for chat_id, symbol, threshold, last_direction in entries:
        ticker = tickers.get(symbol)
        if ticker is None:
            LOG.warning("Ticker not returned for watched symbol %s", symbol)
            continue
        last_price, change_24h_pct = ticker
        direction, next_state = evaluate_price_alert(
            change_24h_pct, threshold, last_direction
        )
        if direction is None and next_state != last_direction:
            store.set_watch_alert_direction(chat_id, symbol, next_state)
        if direction is None:
            continue

        movement = "выросла" if direction == "up" else "упала"
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔔 АЛЕРТ • {symbol}\n"
                    f"Цена {movement} на {change_24h_pct:+.2f}% за 24 часа.\n"
                    f"Текущая цена: {format_price(last_price)} USDT.\n"
                    f"Ваш порог: ±{threshold:g}%.\n"
                    "Это уведомление о движении рынка, не торговая рекомендация."
                ),
            )
        except TelegramError:
            LOG.exception("Failed to send price alert for %s to chat %s", symbol, chat_id)
        else:
            store.set_watch_alert_direction(chat_id, symbol, next_state)


async def get_market_symbols(
    bot_data: dict,
    config: Config,
    client: BybitClient,
) -> tuple[str, ...]:
    if not config.all_usdt_perpetuals:
        return config.symbols

    active_symbols = bot_data.get("active_symbols")
    refreshed_at = bot_data.get("symbols_refreshed_at", 0.0)
    if not active_symbols or time.monotonic() - refreshed_at >= SYMBOL_CACHE_SECONDS:
        active_symbols = await client.get_active_usdt_perpetual_symbols()
        bot_data["active_symbols"] = active_symbols
        bot_data["symbols_refreshed_at"] = time.monotonic()
        LOG.info("Loaded %d active USDT perpetual symbols from Bybit", len(active_symbols))
    return active_symbols


def subscription_batches(topics: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_length = 0
    for topic in topics:
        topic_length = len(topic) + 3
        if topic_length > MAX_SUBSCRIPTION_ARGUMENT_CHARS:
            raise ValueError(f"WebSocket topic is too long: {topic}")
        if current and current_length + topic_length > MAX_SUBSCRIPTION_ARGUMENT_CHARS:
            batches.append(current)
            current = []
            current_length = 0
        current.append(topic)
        current_length += topic_length
    if current:
        batches.append(current)
    return batches


async def seed_interval_history(
    application: Application,
    interval: str,
    symbols: tuple[str, ...],
) -> None:
    config: Config = application.bot_data["config"]
    client: BybitClient = application.bot_data["bybit"]
    cache_by_interval = application.bot_data.setdefault("candle_cache", {})
    candles_by_symbol = cache_by_interval.setdefault(interval, {})
    period = max(
        config.slow_ema_period,
        config.rsi_period + 1,
        config.atr_period + 1,
        config.volume_lookback + 1,
    )
    if interval == config.trend_interval:
        period = max(period, config.trend_ema_period + 1)
    limit = min(1000, max(200, period + 2))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOL_SCANS)
    completed = 0

    async def seed_symbol(symbol: str) -> None:
        nonlocal completed
        if symbol in candles_by_symbol:
            return
        try:
            async with semaphore:
                candles = await client.get_closed_candles(symbol, interval, limit=limit)
            if candles:
                candles_by_symbol[symbol] = candles
            else:
                LOG.warning("No initial %s candles available for %s", interval, symbol)
        except (httpx.HTTPError, RuntimeError, ValueError):
            LOG.exception("Failed to load initial %s candles for %s", interval, symbol)
        finally:
            completed += 1
            if completed % 100 == 0 or completed == len(symbols):
                LOG.info("Initialized %s candle history for %d/%d markets", interval, completed, len(symbols))

    await asyncio.gather(*(seed_symbol(symbol) for symbol in symbols))


def parse_confirmed_candle(payload: dict) -> tuple[str, Candle] | None:
    topic = payload.get("topic")
    data = payload.get("data")
    if not isinstance(topic, str) or not isinstance(data, list):
        return None
    topic_parts = topic.split(".", 2)
    if len(topic_parts) != 3 or topic_parts[0] != "kline":
        return None
    if not data or not isinstance(data[0], dict) or data[0].get("confirm") is not True:
        return None

    candle_data = data[0]
    try:
        candle = Candle(
            timestamp=int(candle_data["start"]),
            open=float(candle_data["open"]),
            high=float(candle_data["high"]),
            low=float(candle_data["low"]),
            close=float(candle_data["close"]),
            volume=float(candle_data["volume"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Malformed Bybit WebSocket kline: {topic}") from error
    return topic_parts[2], candle


async def handle_closed_candle(application: Application, interval: str, symbol: str, candle: Candle) -> None:
    config: Config = application.bot_data["config"]
    cache_by_interval = application.bot_data.setdefault("candle_cache", {})
    candles_by_symbol = cache_by_interval.setdefault(interval, {})
    candles = candles_by_symbol.setdefault(symbol, [])
    if candles and candle.timestamp < candles[-1].timestamp:
        return
    if candles and candle.timestamp == candles[-1].timestamp:
        candles[-1] = candle
    else:
        candles.append(candle)

    keep_count = max(
        config.slow_ema_period,
        config.trend_ema_period if interval == config.trend_interval else 0,
        config.rsi_period + 1,
        config.atr_period + 1,
        config.volume_lookback + 1,
    ) + 5
    if len(candles) > keep_count:
        del candles[:-keep_count]
    application.bot_data["last_market_event_at"] = datetime.now(MOSCOW_TIMEZONE).strftime(
        "%Y-%m-%d %H:%M:%S МСК"
    )

    if interval != config.interval:
        return

    store: SignalStore = application.bot_data["store"]
    subscribers = store.subscribers()
    latest_rsi_state = detect_rsi_extreme(
        [item.close for item in candles], config.rsi_period
    )
    previous_rsi_state = store.rsi_extreme_state(symbol, interval)
    previous_timestamp, previous_label = (
        previous_rsi_state if previous_rsi_state else (None, "neutral")
    )
    if previous_timestamp is None or candle.timestamp > previous_timestamp:
        current_label = latest_rsi_state[0] if latest_rsi_state else "neutral"
        if current_label != "neutral" and current_label != previous_label:
            message = format_rsi_extreme(
                symbol,
                current_label,
                latest_rsi_state[1],
                candle.close,
                candle.timestamp,
                interval,
            )
            for chat_id in subscribers:
                try:
                    await application.bot.send_message(chat_id=chat_id, text=message)
                except TelegramError:
                    LOG.exception(
                        "Failed to deliver RSI extreme alert for %s to chat %s",
                        symbol,
                        chat_id,
                    )
        store.mark_rsi_extreme_processed(symbol, interval, candle.timestamp, current_label)

    last_processed = store.last_processed(symbol, interval)
    if last_processed is not None and candle.timestamp <= last_processed:
        return

    signal = build_signal(
        symbol,
        candles,
        fast_period=config.fast_ema_period,
        slow_period=config.slow_ema_period,
        rsi_period=config.rsi_period,
        atr_period=config.atr_period,
        atr_multiplier=config.atr_multiplier,
        take_profit_1_r=config.take_profit_1_r,
        take_profit_2_r=config.take_profit_2_r,
        volume_multiplier=config.volume_multiplier,
        volume_lookback=config.volume_lookback,
    )
    if signal:
        trend_candles = (
            application.bot_data.get("candle_cache", {})
            .get(config.trend_interval, {})
            .get(symbol, [])
        )
        if confirm_trend(signal, trend_candles, config.trend_ema_period):
            message = format_signal(signal, config)
            for chat_id in subscribers:
                try:
                    await application.bot.send_message(chat_id=chat_id, text=message)
                except TelegramError:
                    LOG.exception("Failed to deliver signal for %s to chat %s", symbol, chat_id)
        else:
            LOG.info(
                "Ignored %s signal for %s: trend EMA %s on interval %s did not confirm",
                signal.direction,
                symbol,
                config.trend_ema_period,
                config.trend_interval,
            )
    store.mark_processed(symbol, interval, candle.timestamp)


async def stream_interval(
    application: Application,
    interval: str,
    symbols: tuple[str, ...],
    refresh_at: float,
) -> None:
    topics = [f"kline.{interval}.{symbol}" for symbol in symbols]
    batches = subscription_batches(topics)
    backoff = 2
    while time.monotonic() < refresh_at:
        try:
            async with websockets.connect(
                BYBIT_LINEAR_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=4 * 1024 * 1024,
                close_timeout=5,
            ) as websocket:
                for batch in batches:
                    await websocket.send(json.dumps({"op": "subscribe", "args": batch}))
                    await asyncio.sleep(0.1)
                application.bot_data["market_stream_status"] = "connected"
                LOG.info("Subscribed to %d Bybit %s candle topics", len(topics), interval)
                backoff = 2

                while time.monotonic() < refresh_at:
                    try:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=20)
                    except asyncio.TimeoutError:
                        await websocket.send(json.dumps({"op": "ping"}))
                        continue

                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        continue
                    if message.get("success") is False:
                        raise RuntimeError(
                            f"Bybit WebSocket subscription failed: {message.get('ret_msg')}"
                        )
                    confirmed = parse_confirmed_candle(message)
                    if confirmed:
                        symbol, candle = confirmed
                        await handle_closed_candle(application, interval, symbol, candle)
        except (
            OSError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            WebSocketException,
        ):
            application.bot_data["market_stream_status"] = "reconnecting"
            LOG.exception("Bybit %s candle stream disconnected; reconnecting", interval)
            await asyncio.sleep(min(backoff, max(0, refresh_at - time.monotonic())))
            backoff = min(backoff * 2, 30)


async def run_market_streams(application: Application) -> None:
    config: Config = application.bot_data["config"]
    client: BybitClient = application.bot_data["bybit"]
    intervals = tuple(dict.fromkeys((config.interval, config.trend_interval)))

    while True:
        application.bot_data["market_stream_status"] = "initializing"
        try:
            symbols = await get_market_symbols(application.bot_data, config, client)
            for interval in intervals:
                await seed_interval_history(application, interval, symbols)
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError):
            application.bot_data["market_stream_status"] = "reconnecting"
            LOG.exception("Unable to initialize Bybit market streams")
            await asyncio.sleep(10)
            continue

        application.bot_data["market_stream_status"] = "connecting"
        LOG.info(
            "Starting real-time Bybit candle feeds for %d markets at intervals %s",
            len(symbols),
            intervals,
        )
        refresh_at = time.monotonic() + WS_SYMBOL_REFRESH_SECONDS
        await asyncio.gather(
            *(
                stream_interval(application, interval, symbols, refresh_at)
                for interval in intervals
            )
        )


async def post_init(application: Application) -> None:
    application.job_queue.scheduler.timezone = MOSCOW_TIMEZONE
    application.bot_data["http_client"] = httpx.AsyncClient(
        timeout=BYBIT_HTTP_TIMEOUT_SECONDS
    )
    application.bot_data["bybit"] = BybitClient(application.bot_data["http_client"])
    application.create_task(
        run_market_streams(application),
        name="bybit-market-streams",
    )
    application.job_queue.run_repeating(
        check_price_alerts,
        interval=PRICE_ALERT_SCAN_SECONDS,
        first=10,
        name="watchlist-price-alerts",
        job_kwargs={"max_instances": 1, "coalesce": True},
    )


async def post_shutdown(application: Application) -> None:
    http_client: httpx.AsyncClient = application.bot_data["http_client"]
    await http_client.aclose()
    store: SignalStore = application.bot_data["store"]
    store.close()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = Config.from_env()
    store = SignalStore(config.database_path)
    application = (
        ApplicationBuilder()
        .token(config.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data["config"] = config
    application.bot_data["store"] = store
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("symbols", symbols))
    application.add_handler(CommandHandler("watch", watch))
    application.add_handler(CommandHandler("unwatch", unwatch))
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    markets = "all active USDT perpetuals" if config.all_usdt_perpetuals else config.symbols
    LOG.info("Starting signal-only bot for %s at interval %s", markets, config.interval)
    application.run_polling()


if __name__ == "__main__":
    main()
