from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from bot import BybitClient
from strategy import (
    Candle,
    Signal,
    atr,
    build_ema_ribbon_pullback_signals,
    build_signal,
    confirm_ema_ribbon,
    confirm_trend,
    ema,
    rsi,
)

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
FILTER_VARIANTS = (
    (
        f"EMA{TREND_EMA_PERIOD} + volume {VOLUME_MULTIPLIER:g}x",
        TREND_EMA_PERIOD,
        VOLUME_MULTIPLIER,
    ),
    ("EMA200 + volume 1.5x", 200, 1.5),
    ("EMA200 + volume 2.0x", 200, 2.0),
    ("EMA300 + volume 1.2x", 300, 1.2),
    ("EMA300 + volume 1.5x", 300, 1.5),
    ("EMA20/50/100/200 ribbon + volume 1.2x", 0, 1.2),
)
EMA_RIBBON_PULLBACK_NAME = "EMA ribbon 20/50/100/200 pullback + volume 1.2x"
CURRENT_FILTER_NAME = FILTER_VARIANTS[0][0]
MIN_RESOLVED_FOR_SELECTION = 200
MIN_VALIDATION_IMPROVEMENT_PP = 2.0
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

    @property
    def resolved(self) -> int:
        return self.tp1_first + self.stop_first

    @property
    def win_rate(self) -> float:
        return 100 * self.tp1_first / self.resolved if self.resolved else 0


def new_window_counts() -> dict[str, OutcomeCounts]:
    return {"train": OutcomeCounts(), "validation": OutcomeCounts()}


def merge_counts(target: OutcomeCounts, source: OutcomeCounts) -> None:
    target.signals += source.signals
    target.tp1_first += source.tp1_first
    target.stop_first += source.stop_first
    target.unresolved += source.unresolved
    target.pending += source.pending


def select_filter_variant(
    training: dict[str, OutcomeCounts],
    validation: dict[str, OutcomeCounts],
) -> str | None:
    if (
        training[CURRENT_FILTER_NAME].resolved < MIN_RESOLVED_FOR_SELECTION
        or validation[CURRENT_FILTER_NAME].resolved < MIN_RESOLVED_FOR_SELECTION
    ):
        return None
    eligible = [
        name
        for name, counts in training.items()
        if counts.resolved >= MIN_RESOLVED_FOR_SELECTION
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda name: training[name].win_rate)
    if best == CURRENT_FILTER_NAME:
        return best
    if (
        validation[best].resolved < MIN_RESOLVED_FOR_SELECTION
        or validation[best].win_rate
        < validation[CURRENT_FILTER_NAME].win_rate + MIN_VALIDATION_IMPROVEMENT_PP
    ):
        return CURRENT_FILTER_NAME
    return best


def ema_ribbon_candidate_passes_validation(
    baseline_train: OutcomeCounts,
    baseline_validation: OutcomeCounts,
    candidate_train: OutcomeCounts,
    candidate_validation: OutcomeCounts,
) -> bool:
    return (
        baseline_train.resolved >= MIN_RESOLVED_FOR_SELECTION
        and baseline_validation.resolved >= MIN_RESOLVED_FOR_SELECTION
        and candidate_train.resolved >= MIN_RESOLVED_FOR_SELECTION
        and candidate_validation.resolved >= MIN_RESOLVED_FOR_SELECTION
        and candidate_train.win_rate > baseline_train.win_rate
        and candidate_validation.win_rate
        >= baseline_validation.win_rate + MIN_VALIDATION_IMPROVEMENT_PP
    )


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
) -> list[tuple[int, Signal, dict[str, Signal | None]]]:
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
        signals_by_volume = {
            volume_multiplier: build_signal(
                "BACKTEST",
                history,
                volume_multiplier=volume_multiplier,
                volume_lookback=VOLUME_LOOKBACK,
            )
            for volume_multiplier in {variant[2] for variant in FILTER_VARIANTS}
        }
        if volatility[atr_index] <= 0:
            continue
        filtered = {
            name: signals_by_volume[volume_multiplier]
            for name, _, volume_multiplier in FILTER_VARIANTS
        }
        candidates.append((index, baseline, filtered))
    return candidates


def find_ema_ribbon_pullback_signals(
    symbol: str, candles: list[Candle], start_ms: int, end_ms: int
) -> list[tuple[int, Signal]]:
    signals = build_ema_ribbon_pullback_signals(symbol, candles)
    index_by_timestamp = {candle.timestamp: index for index, candle in enumerate(candles)}
    return [
        (index_by_timestamp[signal.timestamp], signal)
        for signal in signals
        if start_ms <= signal.timestamp <= end_ms
    ]


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
    split_ms: int,
) -> tuple[
    dict[str, OutcomeCounts],
    dict[str, dict[str, OutcomeCounts]],
    dict[str, OutcomeCounts],
]:
    baseline_counts = new_window_counts()
    filtered_counts = {
        name: new_window_counts() for name, _, _ in FILTER_VARIANTS
    }
    candidates = find_signal_candidates(candles, start_ms, end_ms)
    trend_periods = {period for _, period, _ in FILTER_VARIANTS if period > 0}
    trend_steps = {
        period: interval_milliseconds(TREND_INTERVAL) for period in trend_periods
    }
    trend_indices = {period: -1 for period in trend_periods}
    ribbon_pullback_counts = new_window_counts()

    for candle_index, baseline_signal, signals_by_variant in candidates:
        signal_time = candles[candle_index].timestamp
        if signal_time < split_ms - OUTCOME_HORIZON_BARS * interval_milliseconds(BASE_INTERVAL):
            window = "train"
        elif signal_time >= split_ms:
            window = "validation"
        else:
            continue
        future = candles[candle_index + 1 :]
        candle_close_time = candles[candle_index].timestamp + interval_milliseconds(BASE_INTERVAL)
        baseline_counts[window].add(evaluate_outcome(baseline_signal, future))

        for name, trend_period, _ in FILTER_VARIANTS:
            signal = signals_by_variant[name]
            if signal is None:
                continue
            if trend_period == 0:
                if confirm_ema_ribbon(signal, candles[: candle_index + 1]):
                    filtered_counts[name][window].add(evaluate_outcome(signal, future))
                continue
            trend_index = trend_indices[trend_period]
            trend_step = trend_steps[trend_period]
            while (
                trend_index + 1 < len(trend_candles)
                and trend_candles[trend_index + 1].timestamp + trend_step
                <= candle_close_time
            ):
                trend_index += 1
            trend_indices[trend_period] = trend_index
            if trend_index < 0:
                continue
            if confirm_trend(
                signal, trend_candles[: trend_index + 1], trend_period
            ):
                filtered_counts[name][window].add(evaluate_outcome(signal, future))

    for candle_index, signal in find_ema_ribbon_pullback_signals(
        symbol, candles, start_ms, end_ms
    ):
        signal_time = signal.timestamp
        if signal_time < split_ms - OUTCOME_HORIZON_BARS * interval_milliseconds(BASE_INTERVAL):
            window = "train"
        elif signal_time >= split_ms:
            window = "validation"
        else:
            continue
        ribbon_pullback_counts[window].add(
            evaluate_outcome(signal, candles[candle_index + 1 :])
        )

    return baseline_counts, filtered_counts, ribbon_pullback_counts


def print_counts(label: str, counts: OutcomeCounts) -> None:
    print(
        f"{label}: сигналы={counts.signals}, TP1 раньше стопа={counts.tp1_first}, "
        f"стоп раньше TP1={counts.stop_first}, "
        f"процент TP1 среди разрешенных={counts.win_rate:.1f}% "
        f"(разрешенных исходов={counts.resolved}), "
        f"не достигли уровня за 24ч={counts.unresolved}, "
        f"ожидают исхода={counts.pending}"
    )


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=30) as session:
        client = BybitClient(session)
        if args.top_by_turnover:
            symbols = await client.get_top_usdt_perpetual_symbols_by_turnover(
                args.top_by_turnover
            )
        elif args.symbols.upper() == "ALL":
            symbols = await client.get_active_usdt_perpetual_symbols()
        else:
            symbols = tuple(
                dict.fromkeys(
                    part.strip().upper()
                    for part in args.symbols.split(",")
                    if part.strip()
                )
            )
        if not symbols:
            raise ValueError("No symbols were selected")

        now = datetime.now(timezone.utc)
        end_ms = int(now.timestamp() * 1000)
        requested_start_ms = int((now - timedelta(days=args.days)).timestamp() * 1000)
        base_warmup_ms = (
            max(SLOW_EMA_PERIOD, 200) + VOLUME_LOOKBACK + 2
        ) * interval_milliseconds(BASE_INTERVAL)
        max_trend_period = max(
            period for _, period, _ in FILTER_VARIANTS if period > 0
        )
        trend_warmup_ms = (max_trend_period + 2) * interval_milliseconds(TREND_INTERVAL)
        base_start_ms = requested_start_ms - base_warmup_ms
        trend_start_ms = requested_start_ms - trend_warmup_ms
        split_ms = requested_start_ms + (end_ms - requested_start_ms) * 2 // 3

        baseline_total = new_window_counts()
        filtered_total = {
            name: new_window_counts() for name, _, _ in FILTER_VARIANTS
        }
        ribbon_pullback_total = new_window_counts()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
        completed = 0
        errors = 0

        async def scan_symbol(
            symbol: str,
        ) -> tuple[
            dict[str, OutcomeCounts],
            dict[str, dict[str, OutcomeCounts]],
            dict[str, OutcomeCounts],
        ] | None:
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
                    split_ms,
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
            baseline, filtered, ribbon_pullback = result
            for window in ("train", "validation"):
                merge_counts(baseline_total[window], baseline[window])
                merge_counts(ribbon_pullback_total[window], ribbon_pullback[window])
                for name, _, _ in FILTER_VARIANTS:
                    merge_counts(
                        filtered_total[name][window], filtered[name][window]
                    )

    print(
        f"\nИстория: {args.days} дней, базовый ТФ {BASE_INTERVAL}м, "
        f"сигналы разделены по времени: первые 2/3 для отбора, последняя 1/3 для проверки. "
        "Фильтры EMA на 60м, EMA-лента и объем сравниваются; возврат к EMA20 тестируется отдельно."
    )
    market_selection = (
        f"топ {args.top_by_turnover} по обороту"
        if args.top_by_turnover
        else "все выбранные"
    )
    print(f"Рынков: {len(symbols)} ({market_selection}), ошибок загрузки: {errors}")
    for window in ("train", "validation"):
        print_counts(f"Без фильтров / {window}", baseline_total[window])
    for name, _, _ in FILTER_VARIANTS:
        for window in ("train", "validation"):
            print_counts(f"{name} / {window}", filtered_total[name][window])
    for window in ("train", "validation"):
        print_counts(
            f"{EMA_RIBBON_PULLBACK_NAME} / {window}",
            ribbon_pullback_total[window],
        )
    selected = select_filter_variant(
        {name: counts["train"] for name, counts in filtered_total.items()},
        {name: counts["validation"] for name, counts in filtered_total.items()},
    )
    if selected is None:
        print(
            "Рекомендация: данных недостаточно для выбора фильтра "
            f"(нужно не менее {MIN_RESOLVED_FOR_SELECTION} TP1/стоп исходов на обучающем участке)."
        )
    elif selected == CURRENT_FILTER_NAME:
        print(
            "Рекомендация: оставить текущий фильтр; более строгий вариант не показал "
            f"улучшение не менее чем на {MIN_VALIDATION_IMPROVEMENT_PP:.0f} п.п. "
            "на проверочном участке с достаточным числом исходов."
        )
    else:
        print(
            f"Кандидат для проверки/внедрения: {selected}. "
            "Он отобран на первых 2/3 истории и показал лучший win rate на последней 1/3; "
            "это не гарантия будущего результата."
        )
    if ema_ribbon_candidate_passes_validation(
        filtered_total[CURRENT_FILTER_NAME]["train"],
        filtered_total[CURRENT_FILTER_NAME]["validation"],
        ribbon_pullback_total["train"],
        ribbon_pullback_total["validation"],
    ):
        print(
            f"EMA pullback-кандидат {EMA_RIBBON_PULLBACK_NAME} прошел текущий "
            "критерий отбора/проверки. Перед включением в бота требуется отдельное "
            "решение; проверка не гарантирует будущую прибыль."
        )
    else:
        print(
            "EMA pullback-стратегия не прошла критерий: для отбора нужны минимум "
            f"{MIN_RESOLVED_FOR_SELECTION} TP1/стоп исходов в train и validation, "
            f"улучшение train и не менее {MIN_VALIDATION_IMPROVEMENT_PP:.0f} п.п. "
            "на контрольной выборке относительно текущего фильтра."
        )
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
    parser.add_argument(
        "--top-by-turnover",
        type=int,
        help="Limit ALL to the top N active USDT perpetual markets by 24-hour turnover",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback period in days (1-365)")
    args = parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be between 1 and 365")
    if args.top_by_turnover is not None and args.top_by_turnover < 1:
        parser.error("--top-by-turnover must be positive")
    if args.top_by_turnover is not None and args.symbols.upper() != "ALL":
        parser.error("--top-by-turnover can only be used with --symbols ALL")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
