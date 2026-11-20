from __future__ import annotations

import asyncio
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from strategy import Candle, Signal, build_signal

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info"
BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"
VALID_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
SYMBOL_CACHE_SECONDS = 60 * 60
MAX_CONCURRENT_SYMBOL_SCANS = 8
BYBIT_REQUEST_INTERVAL_SECONDS = 0.12
BYBIT_HTTP_TIMEOUT_SECONDS = 30
PRICE_ALERT_SCAN_SECONDS = 60
DEFAULT_PRICE_ALERT_THRESHOLD = 5.0
MIN_PRICE_ALERT_THRESHOLD = 0.1
MAX_PRICE_ALERT_THRESHOLD = 1000.0
LOG = logging.getLogger("bybit_signals")


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    symbols: tuple[str, ...]
    all_usdt_perpetuals: bool
    interval: str
    scan_interval_seconds: int
    database_path: str
    fast_ema_period: int
    slow_ema_period: int
    rsi_period: int
    atr_period: int
    atr_multiplier: float
    take_profit_1_r: float
    take_profit_2_r: float

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
            default_scan_interval = "300" if all_usdt_perpetuals else "60"
            scan_interval = int(
                os.environ.get("SCAN_INTERVAL_SECONDS", default_scan_interval)
            )
        except ValueError as error:
            raise ValueError("SCAN_INTERVAL_SECONDS must be an integer") from error
        if scan_interval < 30:
            raise ValueError("SCAN_INTERVAL_SECONDS must be at least 30")

        try:
            fast_ema = int(os.environ.get("FAST_EMA_PERIOD", "20"))
            slow_ema = int(os.environ.get("SLOW_EMA_PERIOD", "50"))
            rsi_period = int(os.environ.get("RSI_PERIOD", "14"))
            atr_period = int(os.environ.get("ATR_PERIOD", "14"))
            atr_multiplier = float(os.environ.get("ATR_MULTIPLIER", "1.5"))
            take_profit_1_r = float(os.environ.get("TAKE_PROFIT_1_R", "1"))
            take_profit_2_r = float(os.environ.get("TAKE_PROFIT_2_R", "2"))
        except ValueError as error:
            raise ValueError("Strategy periods and multipliers must be numeric") from error
        if fast_ema < 2 or slow_ema <= fast_ema or slow_ema > 198:
            raise ValueError(
                "EMA periods must satisfy 2 <= FAST_EMA_PERIOD < SLOW_EMA_PERIOD <= 198"
            )
        if rsi_period < 2 or atr_period < 1:
            raise ValueError("RSI_PERIOD must be at least 2 and ATR_PERIOD must be positive")
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
            scan_interval_seconds=scan_interval,
            database_path=os.environ.get("DATABASE_PATH", "signals.sqlite3"),
            fast_ema_period=fast_ema,
            slow_ema_period=slow_ema,
            rsi_period=rsi_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            take_profit_1_r=take_profit_1_r,
            take_profit_2_r=take_profit_2_r,
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

    async def get_closed_candles(self, symbol: str, interval: str) -> list[Candle]:
        payload = await self._get_payload(
            BYBIT_KLINE_URL,
            {"category": "linear", "symbol": symbol, "interval": interval, "limit": 200},
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
                )
                for row in closed_rows
            ]
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Bybit returned malformed kline data for {symbol}") from error

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
    opened_at = datetime.fromtimestamp(signal.timestamp / 1000, tz=timezone.utc)
    interval_label = {"D": "1д", "W": "1н", "M": "1м"}.get(config.interval, f"{config.interval}м")
    return (
        f"🚨 СИГНАЛ • {signal.direction} {direction_emoji}\n"
        f"#{signal.symbol} · Bybit Perpetual · {interval_label}\n"
        f"Время свечи: {opened_at:%Y-%m-%d %H:%M UTC}\n\n"
        f"💰 Вход: {format_price(signal.entry)}\n"
        f"🎯 Тейк 1: {format_price(signal.take_profit_1)} "
        f"({price_change(signal.entry, signal.take_profit_1, signal.direction):+.2f}%)\n"
        f"🎯 Тейк 2: {format_price(signal.take_profit_2)} "
        f"({price_change(signal.entry, signal.take_profit_2, signal.direction):+.2f}%)\n"
        f"🛑 Стоп: {format_price(signal.stop_loss)} "
        f"({price_change(signal.entry, signal.stop_loss, signal.direction):+.2f}%)\n\n"
        f"Сигнал по EMA {config.fast_ema_period}/{config.slow_ema_period}, подтверждение RSI "
        f"{config.rsi_period}; уровни рассчитаны по ATR {config.atr_period}.\n"
        "Только аналитическая информация: бот не торгует и не является финансовой рекомендацией."
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
            "с подтверждением RSI. Уровни рассчитываются по ATR.\n"
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
    last_scan = context.application.bot_data.get("last_scan_at", "проверка еще не выполнялась")
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
        f"RSI {config.rsi_period}, ATR {config.atr_period} x{config.atr_multiplier:g}\n"
        f"Тейки: {config.take_profit_1_r:g}R / {config.take_profit_2_r:g}R\n"
        f"Интервал сканирования: {config.scan_interval_seconds} сек.\n"
        f"Последняя проверка: {last_scan}"
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    config: Config = context.application.bot_data["config"]
    active_symbols = context.application.bot_data.get("active_symbols", ())
    if config.all_usdt_perpetuals:
        if not active_symbols:
            await update.message.reply_text(
                "Список активных USDT perpetual-контрактов Bybit загрузится при первом сканировании."
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


async def get_scan_symbols(
    context: ContextTypes.DEFAULT_TYPE,
    config: Config,
    client: BybitClient,
) -> tuple[str, ...]:
    if not config.all_usdt_perpetuals:
        return config.symbols

    bot_data = context.application.bot_data
    active_symbols = bot_data.get("active_symbols")
    refreshed_at = bot_data.get("symbols_refreshed_at", 0.0)
    if not active_symbols or time.monotonic() - refreshed_at >= SYMBOL_CACHE_SECONDS:
        active_symbols = await client.get_active_usdt_perpetual_symbols()
        bot_data["active_symbols"] = active_symbols
        bot_data["symbols_refreshed_at"] = time.monotonic()
        LOG.info("Loaded %d active USDT perpetual symbols from Bybit", len(active_symbols))
    return active_symbols


async def scan_market(context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    store: SignalStore = context.application.bot_data["store"]
    client: BybitClient = context.application.bot_data["bybit"]
    context.application.bot_data["last_scan_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    if config.all_usdt_perpetuals:
        try:
            scan_symbols = await get_scan_symbols(context, config, client)
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError):
            LOG.exception("Failed to load active USDT perpetual symbols")
            return
    else:
        scan_symbols = config.symbols

    subscribers = store.subscribers()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOL_SCANS)

    async def scan_symbol(symbol: str) -> None:
        try:
            async with semaphore:
                candles = await client.get_closed_candles(symbol, config.interval)
            if not candles:
                LOG.warning("Not enough closed candles for %s", symbol)
                return
            latest = candles[-1]
            last_processed = store.last_processed(symbol, config.interval)
            if last_processed is not None and latest.timestamp <= last_processed:
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
            )
            if signal:
                message = format_signal(signal, config)
                for chat_id in subscribers:
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=message)
                    except TelegramError:
                        LOG.exception("Failed to deliver signal for %s to chat %s", symbol, chat_id)
            store.mark_processed(symbol, config.interval, latest.timestamp)
        except (httpx.HTTPError, RuntimeError, ValueError):
            LOG.exception("Failed to scan %s", symbol)

    await asyncio.gather(*(scan_symbol(symbol) for symbol in scan_symbols))


async def post_init(application: Application) -> None:
    application.bot_data["http_client"] = httpx.AsyncClient(
        timeout=BYBIT_HTTP_TIMEOUT_SECONDS
    )
    application.bot_data["bybit"] = BybitClient(application.bot_data["http_client"])
    config: Config = application.bot_data["config"]
    application.job_queue.run_repeating(
        scan_market,
        interval=config.scan_interval_seconds,
        first=5,
        name="bybit-market-scan",
        job_kwargs={"max_instances": 1, "coalesce": True},
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
