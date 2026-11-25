from __future__ import annotations

from dataclasses import dataclass

EMA_PULLBACK_STRATEGY_NAME = "EMA PULLBACK 20/50/200 CONFLUENCE"


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: str
    entry: float
    take_profit_1: float
    take_profit_2: float
    stop_loss: float
    timestamp: int
    strategy: str = "EMA 20/50 crossover"


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


def detect_rsi_extreme(
    values: list[float],
    period: int = 14,
    overbought: float = 70,
    oversold: float = 30,
) -> tuple[str, float] | None:
    readings = rsi(values, period)
    if not readings:
        return None

    current = readings[-1]
    if current >= overbought:
        return "overbought", current
    if current <= oversold:
        return "oversold", current
    return None


def confirm_trend(signal: Signal, candles: list[Candle], ema_period: int = 200) -> bool:
    trend = ema([candle.close for candle in candles], ema_period)
    if not trend:
        return False
    trend_ema = trend[-1]
    latest_close = candles[-1].close
    if signal.direction == "LONG":
        return latest_close > trend_ema
    if signal.direction == "SHORT":
        return latest_close < trend_ema
    return False


def confirm_ema_ribbon(signal: Signal, candles: list[Candle]) -> bool:
    closes = [candle.close for candle in candles]
    lines = {period: ema(closes, period) for period in (20, 50, 100, 200)}
    if any(not values for values in lines.values()):
        return False
    levels = [lines[period][-1] for period in (20, 50, 100, 200)]
    if signal.direction == "LONG":
        return levels[0] > levels[1] > levels[2] > levels[3] and closes[-1] > levels[0]
    if signal.direction == "SHORT":
        return levels[0] < levels[1] < levels[2] < levels[3] and closes[-1] < levels[0]
    return False


def build_ema_pullback_signal(
    symbol: str,
    candles: list[Candle],
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    take_profit_1_r: float = 1,
    take_profit_2_r: float = 2,
    pullback_lookback: int = 3,
    rsi_period: int = 14,
    volume_multiplier: float = 1.2,
    volume_lookback: int = 20,
) -> Signal | None:
    if pullback_lookback < 1:
        raise ValueError("pullback_lookback must be positive")
    if len(candles) < 201 + pullback_lookback:
        return None

    closes = [candle.close for candle in candles]
    lines = {period: ema(closes, period) for period in (20, 50, 200)}
    volatility = atr(candles, atr_period)
    momentum = rsi(closes, rsi_period)
    if (
        any(not values for values in lines.values())
        or not volatility
        or not momentum
        or len(candles) <= volume_lookback
    ):
        return None

    current_ema20 = lines[20][-1]
    current_ema50 = lines[50][-1]
    current_ema200 = lines[200][-1]
    current = candles[-1]
    if current.close <= 0:
        return None

    if current.close > current_ema200 and current_ema20 > current_ema50 > current_ema200:
        direction = "LONG"
    elif current.close < current_ema200 and current_ema20 < current_ema50 < current_ema200:
        direction = "SHORT"
    else:
        return None

    current_rsi = momentum[-1]
    if direction == "LONG" and not 50 <= current_rsi <= 65:
        return None
    if direction == "SHORT" and not 35 <= current_rsi <= 50:
        return None

    average_volume = (
        sum(candle.volume for candle in candles[-volume_lookback - 1 : -1])
        / volume_lookback
    )
    if average_volume <= 0 or candles[-1].volume < average_volume * volume_multiplier:
        return None

    touched_levels = []
    for index in range(len(candles) - pullback_lookback - 1, len(candles) - 1):
        candle = candles[index]
        for period in (20, 50):
            level = lines[period][index - (period - 1)]
            if candle.low <= level <= candle.high:
                touched_levels.append(level)
    if not touched_levels:
        return None

    previous = candles[-2]
    body = abs(current.close - current.open)
    upper_wick = current.high - max(current.open, current.close)
    lower_wick = min(current.open, current.close) - current.low
    candle_range = current.high - current.low
    bullish_engulfing = (
        previous.close < previous.open
        and current.close > current.open
        and current.open <= previous.close
        and current.close >= previous.open
    )
    bearish_engulfing = (
        previous.close > previous.open
        and current.close < current.open
        and current.open >= previous.close
        and current.close <= previous.open
    )
    bullish_pin_bar = (
        body > 0
        and candle_range > 0
        and lower_wick >= 2 * body
        and lower_wick >= 2 * upper_wick
        and (current.close - current.low) / candle_range >= 2 / 3
    )
    bearish_pin_bar = (
        body > 0
        and candle_range > 0
        and upper_wick >= 2 * body
        and upper_wick >= 2 * lower_wick
        and (current.high - current.close) / candle_range >= 2 / 3
    )
    if direction == "LONG":
        if not (bullish_engulfing or bullish_pin_bar) or not any(
            current.close > level for level in touched_levels
        ):
            return None
    elif not (bearish_engulfing or bearish_pin_bar) or not any(
        current.close < level for level in touched_levels
    ):
        return None

    current_atr = volatility[-1]
    risk = atr_multiplier * current_atr
    if current_atr <= 0 or risk >= current.close:
        return None
    if direction == "SHORT" and take_profit_2_r * risk >= current.close:
        return None
    sign = 1 if direction == "LONG" else -1
    return Signal(
        symbol=symbol,
        direction=direction,
        entry=current.close,
        take_profit_1=current.close + sign * take_profit_1_r * risk,
        take_profit_2=current.close + sign * take_profit_2_r * risk,
        stop_loss=current.close - sign * risk,
        timestamp=current.timestamp,
        strategy=EMA_PULLBACK_STRATEGY_NAME,
    )


def build_ema_ribbon_pullback_signals(
    symbol: str,
    candles: list[Candle],
    fast_period: int = 20,
    ribbon_periods: tuple[int, ...] = (20, 50, 100, 200),
    rsi_period: int = 14,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    take_profit_1_r: float = 1,
    take_profit_2_r: float = 2,
    volume_multiplier: float = 1.2,
    volume_lookback: int = 20,
) -> list[Signal]:
    if len(ribbon_periods) < 2 or tuple(sorted(set(ribbon_periods))) != ribbon_periods:
        raise ValueError("ribbon_periods must be unique and in ascending order")
    if fast_period != ribbon_periods[0]:
        raise ValueError("fast_period must be the shortest EMA ribbon period")
    if volume_lookback <= 0 or volume_multiplier <= 0:
        raise ValueError("volume_lookback and volume_multiplier must be positive")

    closes = [candle.close for candle in candles]
    lines = {period: ema(closes, period) for period in ribbon_periods}
    momentum = rsi(closes, rsi_period)
    volatility = atr(candles, atr_period)
    warmup = max(max(ribbon_periods), rsi_period + 1, atr_period + 1, volume_lookback + 1)
    if (
        len(candles) <= warmup
        or any(not values for values in lines.values())
        or not momentum
        or not volatility
    ):
        return []

    signals = []
    for index in range(warmup, len(candles)):
        ema_levels = [
            lines[period][index - (period - 1)] for period in ribbon_periods
        ]
        previous_fast = lines[fast_period][index - (fast_period - 1) - 1]
        current_fast = ema_levels[0]
        previous_close = candles[index - 1].close
        current_close = candles[index].close
        rsi_index = index - rsi_period
        atr_index = index - atr_period
        current_rsi = momentum[rsi_index]
        current_atr = volatility[atr_index]
        average_volume = (
            sum(candle.volume for candle in candles[index - volume_lookback : index])
            / volume_lookback
        )
        if (
            current_atr <= 0
            or average_volume <= 0
            or candles[index].volume < average_volume * volume_multiplier
        ):
            continue

        direction = None
        if (
            all(left > right for left, right in zip(ema_levels, ema_levels[1:]))
            and previous_close <= previous_fast
            and current_close > current_fast
            and 50 <= current_rsi < 70
        ):
            direction = "LONG"
        elif (
            all(left < right for left, right in zip(ema_levels, ema_levels[1:]))
            and previous_close >= previous_fast
            and current_close < current_fast
            and 30 < current_rsi <= 50
        ):
            direction = "SHORT"
        if direction is None:
            continue

        risk = atr_multiplier * current_atr
        if risk >= current_close or (direction == "SHORT" and take_profit_2_r * risk >= current_close):
            continue
        sign = 1 if direction == "LONG" else -1
        signals.append(
            Signal(
                symbol=symbol,
                direction=direction,
                entry=current_close,
                take_profit_1=current_close + sign * take_profit_1_r * risk,
                take_profit_2=current_close + sign * take_profit_2_r * risk,
                stop_loss=current_close - sign * risk,
                timestamp=candles[index].timestamp,
            )
        )
    return signals


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
    volume_multiplier: float | None = 1.2,
    volume_lookback: int = 20,
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
    if volume_multiplier is not None:
        if volume_lookback <= 0 or len(candles) <= volume_lookback:
            return None
        average_volume = (
            sum(candle.volume for candle in candles[-volume_lookback - 1 : -1])
            / volume_lookback
        )
        if average_volume <= 0 or candles[-1].volume < average_volume * volume_multiplier:
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
            strategy=f"EMA {fast_period}/{slow_period} crossover",
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
            strategy=f"EMA {fast_period}/{slow_period} crossover",
        )

    return None


def _rsi_value(average_gain: float, average_loss: float) -> float:
    if average_loss == 0:
        return 100.0 if average_gain else 50.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)
