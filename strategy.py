from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: str
    entry: float
    take_profit_1: float
    take_profit_2: float
    stop_loss: float
    timestamp: int


def ema(values: list[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for value in values[period:]:
        result.append((value - result[-1]) * multiplier + result[-1])
    return result


def rsi(values: list[float], period: int = 14) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) <= period:
        return []

    changes = [current - previous for previous, current in zip(values, values[1:])]
    gains = [max(change, 0) for change in changes]
    losses = [max(-change, 0) for change in changes]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    result = [_rsi_value(average_gain, average_loss)]

    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period
        result.append(_rsi_value(average_gain, average_loss))
    return result


def atr(candles: list[Candle], period: int = 14) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(candles) <= period:
        return []

    true_ranges = []
    for previous, current in zip(candles, candles[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )

    average = sum(true_ranges[:period]) / period
    result = [average]
    for true_range in true_ranges[period:]:
        average = (average * (period - 1) + true_range) / period
        result.append(average)
    return result


def build_signal(
    symbol: str,
    candles: list[Candle],
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    take_profit_1_r: float = 1,
    take_profit_2_r: float = 2,
) -> Signal | None:
    closes = [candle.close for candle in candles]
    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    momentum = rsi(closes, rsi_period)
    volatility = atr(candles, atr_period)

    if min(len(fast), len(slow), len(momentum), len(volatility)) < 2:
        return None

    previous_fast, current_fast = fast[-2:]
    previous_slow = slow[-2]
    current_slow = slow[-1]
    current_rsi = momentum[-1]
    current_atr = volatility[-1]
    entry = closes[-1]
    if entry <= 0 or current_atr <= 0:
        return None

    if previous_fast <= previous_slow and current_fast > current_slow:
        if not 50 <= current_rsi < 70:
            return None
        risk = atr_multiplier * current_atr
        if risk >= entry:
            return None
        return Signal(
            symbol=symbol,
            direction="LONG",
            entry=entry,
            take_profit_1=entry + take_profit_1_r * risk,
            take_profit_2=entry + take_profit_2_r * risk,
            stop_loss=entry - risk,
            timestamp=candles[-1].timestamp,
        )

    if previous_fast >= previous_slow and current_fast < current_slow:
        if not 30 < current_rsi <= 50:
            return None
        risk = atr_multiplier * current_atr
        if take_profit_2_r * risk >= entry:
            return None
        return Signal(
            symbol=symbol,
            direction="SHORT",
            entry=entry,
            take_profit_1=entry - take_profit_1_r * risk,
            take_profit_2=entry - take_profit_2_r * risk,
            stop_loss=entry + risk,
            timestamp=candles[-1].timestamp,
        )

    return None


def _rsi_value(average_gain: float, average_loss: float) -> float:
    if average_loss == 0:
        return 100.0 if average_gain else 50.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)
