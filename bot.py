from __future__ import annotations

import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from strategy import Candle, Signal, build_signal

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
VALID_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
LOG = logging.getLogger("bybit_signals")


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    symbols: tuple[str, ...]
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

        symbols = tuple(
            dict.fromkeys(
                symbol.strip().upper()
                for symbol in os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
                if symbol.strip()
            )
        )
        if not symbols or any(not symbol.endswith("USDT") for symbol in symbols):
            raise ValueError("SYMBOLS must contain one or more USDT symbols, e.g. BTCUSDT,ETHUSDT")

        interval = os.environ.get("INTERVAL", "15").strip().upper()
        if interval not in VALID_INTERVALS:
            raise ValueError(f"INTERVAL must be one of: {', '.join(sorted(VALID_INTERVALS))}")

        try:
            scan_interval = int(os.environ.get("SCAN_INTERVAL_SECONDS", "60"))
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

    def close(self) -> None:
        self.connection.close()


class BybitClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def get_closed_candles(self, symbol: str, interval: str) -> list[Candle]:
        response = await self.client.get(
            BYBIT_KLINE_URL,
            params={"category": "linear", "symbol": symbol, "interval": interval, "limit": 200},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit returned retCode={payload.get('retCode')}: {payload.get('retMsg')}"
            )

        rows = payload.get("result", {}).get("list", [])
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
        "Команды: /status, /symbols, /stop, /help."
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
            "/start — подписаться\n/status — состояние бота\n/symbols — список рынков\n/stop — отписаться"
        )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    config: Config = context.application.bot_data["config"]
    last_scan = context.application.bot_data.get("last_scan_at", "проверка еще не выполнялась")
    await update.message.reply_text(
        f"Режим: сигналы без торговли\n"
        f"Рынки: {', '.join(config.symbols)}\n"
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
    await update.message.reply_text(
        f"Отслеживаемые рынки: {', '.join(config.symbols)}\nТаймфрейм: {config.interval}"
    )


async def scan_market(context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.application.bot_data["config"]
    store: SignalStore = context.application.bot_data["store"]
    client: BybitClient = context.application.bot_data["bybit"]
    context.application.bot_data["last_scan_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    for symbol in config.symbols:
        try:
            candles = await client.get_closed_candles(symbol, config.interval)
            if not candles:
                LOG.warning("Not enough closed candles for %s", symbol)
                continue
            latest = candles[-1]
            last_processed = store.last_processed(symbol, config.interval)
            if last_processed is not None and latest.timestamp <= last_processed:
                continue

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
                for chat_id in store.subscribers():
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=message)
                    except TelegramError:
                        LOG.exception("Failed to deliver signal for %s to chat %s", symbol, chat_id)
            store.mark_processed(symbol, config.interval, latest.timestamp)
        except (httpx.HTTPError, RuntimeError, ValueError):
            LOG.exception("Failed to scan %s", symbol)


async def post_init(application: Application) -> None:
    application.bot_data["http_client"] = httpx.AsyncClient(timeout=10)
    application.bot_data["bybit"] = BybitClient(application.bot_data["http_client"])
    config: Config = application.bot_data["config"]
    application.job_queue.run_repeating(
        scan_market,
        interval=config.scan_interval_seconds,
        first=5,
        name="bybit-market-scan",
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
    LOG.info("Starting signal-only bot for %s at interval %s", config.symbols, config.interval)
    application.run_polling()


if __name__ == "__main__":
    main()
