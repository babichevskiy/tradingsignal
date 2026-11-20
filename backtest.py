from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from bot import BybitClient
from strategy import Candle, Signal, atr, build_signal, confirm_trend, ema, rsi

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
BASE_INTERVAL = "15"
TREND_INTERVAL = "60"
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.2
TREND_EMA_PERIOD = 200
TAKE_PROFIT_1_R = 1.0
TAKE_PROFIT_2_R = 2.0
OUTCOME_HORIZON_BARS = 96
MAX_CONCURRENT_SYMBOLS = 6


@dataclass
class OutcomeCounts:
    signals: int = 0
    tp1_first: int = 0
    stop_first: int = 0
    unresolved: int = 0
    pending: int = 0

    def add(self, result: str) -> None:
        self.signals += 1
        if result == "tp1":
            self.tp1_first += 1
        elif result == "stop":
            self.stop_first += 1
        elif result == "pending":
            self.pending += 1
        else:
            self.unresolved += 1


def interval_milliseconds(interval: str) -> int:
    if interval == "D":
        minutes = 24 * 60
    elif interval == "W":
        minutes = 7 * 24 * 60
    elif interval == "M":
        minutes = 30 * 24 * 60
    else:
        minutes = int(interval)
    return minutes * 60 * 1000


async def fetch_history(
    client: BybitClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    now_ms: int,
) -> list[Candle]:
    step_ms = interval_milliseconds(interval)
    candles_by_time: dict[int, Candle] = {}
    cursor_end = end_ms
    max_pages = math.ceil((end_ms - start_ms) / (step_ms * 1000)) + 2

    for _ in range(max_pages):
        rows = await client.get_historical_candles(
            symbol, interval, start_ms, cursor_end, limit=1000
        )
        if not rows:
            break
        for candle in rows:
            if (
                start_ms <= candle.timestamp
                and candle.timestamp + step_ms <= now_ms
            ):
                candles_by_time[candle.timestamp] = candle

        oldest = min(candle.timestamp for candle in rows)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
    else:
        raise RuntimeError(f"History pagination exceeded the limit for {symbol} {interval}")

    return [candles_by_time[timestamp] for timestamp in sorted(candles_by_time)]


def find_signal_candidates(
    candles: list[Candle],
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, Signal, Signal | None]]:
    closes = [candle.close for candle in candles]
    fast = ema(closes, FAST_EMA_PERIOD)
    slow = ema(closes, SLOW_EMA_PERIOD)
    momentum = rsi(closes, RSI_PERIOD)
    volatility = atr(candles, ATR_PERIOD)
    if not fast or not slow or not momentum or not volatility:
        return []

    candidates = []
    for index in range(SLOW_EMA_PERIOD, len(candles)):
        if not start_ms <= candles[index].timestamp <= end_ms:
            continue

        fast_index = index - (FAST_EMA_PERIOD - 1)
        slow_index = index - (SLOW_EMA_PERIOD - 1)
        rsi_index = index - RSI_PERIOD
        atr_index = index - ATR_PERIOD
        crossed_up = fast[fast_index - 1] <= slow[slow_index - 1] and fast[fast_index] > slow[slow_index]
        crossed_down = fast[fast_index - 1] >= slow[slow_index - 1] and fast[fast_index] < slow[slow_index]
        current_rsi = momentum[rsi_index]
        if not (crossed_up and 50 <= current_rsi < 70) and not (
            crossed_down and 30 < current_rsi <= 50
        ):
            continue

        history = candles[: index + 1]
        baseline = build_signal(
            "BACKTEST",
            history,
            volume_multiplier=None,
            volume_lookback=VOLUME_LOOKBACK,
        )
        if baseline is None:
            continue
        filtered = build_signal(
            "BACKTEST",
            history,
            volume_multiplier=VOLUME_MULTIPLIER,
            volume_lookback=VOLUME_LOOKBACK,
        )
        if volatility[atr_index] <= 0:
            continue
        candidates.append((index, baseline, filtered))
    return candidates


def evaluate_outcome(
    signal: Signal,
    future_candles: list[Candle],
    horizon_bars: int = OUTCOME_HORIZON_BARS,
) -> str:
    for candle in future_candles[:horizon_bars]:
        if signal.direction == "LONG":
            stop_hit = candle.low <= signal.stop_loss
            target_hit = candle.high >= signal.take_profit_1
        else:
            stop_hit = candle.high >= signal.stop_loss
            target_hit = candle.low <= signal.take_profit_1

        if stop_hit:
            return "stop"
        if target_hit:
            return "tp1"

    if len(future_candles) < horizon_bars:
        return "pending"
    return "unresolved"


def run_symbol_backtest(
    symbol: str,
    candles: list[Candle],
    trend_candles: list[Candle],
    start_ms: int,
    end_ms: int,
) -> tuple[OutcomeCounts, OutcomeCounts]:
    baseline_counts = OutcomeCounts()
    filtered_counts = OutcomeCounts()
    candidates = find_signal_candidates(candles, start_ms, end_ms)
    trend_step = interval_milliseconds(TREND_INTERVAL)
    trend_index = -1

    for candle_index, baseline_signal, filtered_signal in candidates:
        future = candles[candle_index + 1 :]
        baseline_counts.add(evaluate_outcome(baseline_signal, future))

        if filtered_signal is None:
            continue
        candle_close_time = candles[candle_index].timestamp + interval_milliseconds(BASE_INTERVAL)
        while (
            trend_index + 1 < len(trend_candles)
            and trend_candles[trend_index + 1].timestamp + trend_step <= candle_close_time
        ):
            trend_index += 1
        if trend_index < 0:
            continue
        if not confirm_trend(
            filtered_signal, trend_candles[: trend_index + 1], TREND_EMA_PERIOD
        ):
            continue
        filtered_counts.add(evaluate_outcome(filtered_signal, future))

    return baseline_counts, filtered_counts


def print_counts(label: str, counts: OutcomeCounts) -> None:
    resolved = counts.tp1_first + counts.stop_first
    win_rate = 100 * counts.tp1_first / resolved if resolved else 0
    print(
        f"{label}: сигналы={counts.signals}, TP1 раньше стопа={counts.tp1_first}, "
        f"стоп раньше TP1={counts.stop_first}, "
        f"процент TP1 среди разрешенных={win_rate:.1f}%, "
        f"не достигли уровня за 24ч={counts.unresolved}, "
        f"ожидают исхода={counts.pending}"
    )


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=30) as session:
        client = BybitClient(session)
        symbols = (
            await client.get_active_usdt_perpetual_symbols()
            if args.symbols.upper() == "ALL"
            else tuple(dict.fromkeys(part.strip().upper() for part in args.symbols.split(",") if part.strip()))
        )
        if not symbols:
            raise ValueError("No symbols were selected")

        now = datetime.now(timezone.utc)
        end_ms = int(now.timestamp() * 1000)
        requested_start_ms = int((now - timedelta(days=args.days)).timestamp() * 1000)
        base_warmup_ms = (SLOW_EMA_PERIOD + VOLUME_LOOKBACK + 2) * interval_milliseconds(BASE_INTERVAL)
        trend_warmup_ms = (TREND_EMA_PERIOD + 2) * interval_milliseconds(TREND_INTERVAL)
        base_start_ms = requested_start_ms - base_warmup_ms
        trend_start_ms = requested_start_ms - trend_warmup_ms

        baseline_total = OutcomeCounts()
        filtered_total = OutcomeCounts()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
        completed = 0
        errors = 0

        async def scan_symbol(symbol: str) -> tuple[OutcomeCounts, OutcomeCounts] | None:
            nonlocal completed, errors
            try:
                async with semaphore:
                    base_history = await fetch_history(
                        client, symbol, BASE_INTERVAL, base_start_ms, end_ms, end_ms
                    )
                    trend_history = await fetch_history(
                        client, symbol, TREND_INTERVAL, trend_start_ms, end_ms, end_ms
                    )
                result = run_symbol_backtest(
                    symbol,
                    base_history,
                    trend_history,
                    requested_start_ms,
                    end_ms,
                )
                return result
            except (httpx.HTTPError, RuntimeError, ValueError):
                errors += 1
                print(f"Ошибка загрузки/теста {symbol}; см. журнал запуска")
                raise
            finally:
                completed += 1
                if completed % 25 == 0 or completed == len(symbols):
                    print(f"Обработано рынков: {completed}/{len(symbols)}")

        tasks = [asyncio.create_task(scan_symbol(symbol)) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(symbols, results):
            if isinstance(result, BaseException):
                print(f"Пропущен рынок {symbol}: {result}")
                continue
            baseline, filtered = result
            baseline_total.signals += baseline.signals
            baseline_total.tp1_first += baseline.tp1_first
            baseline_total.stop_first += baseline.stop_first
            baseline_total.unresolved += baseline.unresolved
            baseline_total.pending += baseline.pending
            filtered_total.signals += filtered.signals
            filtered_total.tp1_first += filtered.tp1_first
            filtered_total.stop_first += filtered.stop_first
            filtered_total.unresolved += filtered.unresolved
            filtered_total.pending += filtered.pending

    print(
        f"\nИстория: {args.days} дней, базовый ТФ {BASE_INTERVAL}м, "
        f"фильтр тренда {TREND_INTERVAL}м EMA {TREND_EMA_PERIOD}, "
        f"фильтр объема {VOLUME_MULTIPLIER}x/{VOLUME_LOOKBACK} свечей."
    )
    print(f"Рынков: {len(symbols)}, ошибок загрузки: {errors}")
    print_counts("Без новых фильтров", baseline_total)
    print_counts("С фильтром EMA тренда и объема", filtered_total)
    print(
        "Исход TP1/стоп оценивается по максимуму 24 часа после сигнала; "
        "если одна свеча касается и TP1, и стопа, считается, что первым сработал стоп. "
        "Комиссии, проскальзывание, funding и перекрывающиеся сигналы не учитываются. "
        "Исторический результат не предсказывает будущую доходность."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical comparison of the Bybit signal filters using public market data."
    )
    parser.add_argument("--symbols", default="ALL", help="ALL or comma-separated USDT symbols")
    parser.add_argument("--days", type=int, default=30, help="Lookback period in days (1-365)")
    args = parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be between 1 and 365")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
