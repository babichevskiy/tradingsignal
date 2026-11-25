from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from bot import BybitClient
from backtest import fetch_history
from strategy import Candle, atr, ema, rsi


@dataclass(frozen=True)
class Variant:
    name: str
    volume_multiplier: float


VARIANTS = (
    Variant("confluence volume 1.2x", 1.2),
    Variant("confluence volume 1.5x", 1.5),
    Variant("confluence volume 2.0x", 2.0),
)


@dataclass
class Counts:
    tp1: int = 0
    stop: int = 0
    unresolved: int = 0

    @property
    def resolved(self) -> int:
        return self.tp1 + self.stop

    @property
    def win_rate(self) -> float:
        return 100 * self.tp1 / self.resolved if self.resolved else 0


def interval_ms(interval: str) -> int:
    return int(interval) * 60 * 1000


def ema_at(values: list[float], index: int, period: int) -> float | None:
    result_index = index - period + 1
    if result_index < 0 or result_index >= len(values):
        return None
    return values[result_index]


def ema_confirmation(direction: str, candles: list[Candle], index: int) -> bool:
    closes = [candle.close for candle in candles]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    level20 = ema_at(ema20, index, 20)
    level50 = ema_at(ema50, index, 50)
    level200 = ema_at(ema200, index, 200)
    if level20 is None or level50 is None or level200 is None:
        return False
    close = candles[index].close
    if direction == "LONG":
        return close > level200 and level20 > level50 > level200
    return close < level200 and level20 < level50 < level200


def bullish_confirmation(candles: list[Candle], index: int) -> bool:
    previous = candles[index - 1]
    current = candles[index]
    body = abs(current.close - current.open)
    candle_range = current.high - current.low
    lower_wick = min(current.open, current.close) - current.low
    upper_wick = current.high - max(current.open, current.close)
    engulfing = (
        previous.close < previous.open
        and current.close > current.open
        and current.open <= previous.close
        and current.close >= previous.open
    )
    pin = (
        body > 0
        and candle_range > 0
        and lower_wick >= 2 * body
        and lower_wick >= 2 * upper_wick
        and (current.close - current.low) / candle_range >= 2 / 3
    )
    return engulfing or pin


def bearish_confirmation(candles: list[Candle], index: int) -> bool:
    previous = candles[index - 1]
    current = candles[index]
    body = abs(current.close - current.open)
    candle_range = current.high - current.low
    lower_wick = min(current.open, current.close) - current.low
    upper_wick = current.high - max(current.open, current.close)
    engulfing = (
        previous.close > previous.open
        and current.close < current.open
        and current.open >= previous.close
        and current.close <= previous.open
    )
    pin = (
        body > 0
        and candle_range > 0
        and upper_wick >= 2 * body
        and upper_wick >= 2 * lower_wick
        and (current.high - current.close) / candle_range >= 2 / 3
    )
    return engulfing or pin


def touch_confirmed(candles: list[Candle], index: int) -> tuple[bool, str | None]:
    closes = [candle.close for candle in candles]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    direction = "LONG" if ema_confirmation("LONG", candles, index) else (
        "SHORT" if ema_confirmation("SHORT", candles, index) else None
    )
    if direction is None:
        return False, None

    touched = False
    ema_touch_levels: list[float] = []
    for touch_index in range(max(200, index - 3), index):
        level20 = ema_at(ema20, touch_index, 20)
        level50 = ema_at(ema50, touch_index, 50)
        candle = candles[touch_index]
        for level in (level20, level50):
            if level is not None and candle.low <= level <= candle.high:
                touched = True
                ema_touch_levels.append(level)

    if not touched:
        return False, None
    close = candles[index].close
    if direction == "LONG":
        return close > min(ema_touch_levels), direction
    return close < max(ema_touch_levels), direction


def evaluate(
    direction: str,
    entry: float,
    risk: float,
    future: list[Candle],
    horizon: int,
) -> str:
    sign = 1 if direction == "LONG" else -1
    stop = entry - sign * risk
    target = entry + sign * risk
    for candle in future[:horizon]:
        if direction == "LONG":
            stop_hit = candle.low <= stop
            target_hit = candle.high >= target
        else:
            stop_hit = candle.high >= stop
            target_hit = candle.low <= target
        if stop_hit:
            return "stop"
        if target_hit:
            return "tp1"
    return "unresolved" if len(future) >= horizon else "pending"


def scan_variant(
    candles: list[Candle],
    variant: Variant,
    interval: str,
) -> tuple[Counts, Counts]:
    closes = [candle.close for candle in candles]
    momentum = rsi(closes, 14)
    volatility = atr(candles, 14)
    train = Counts()
    validation = Counts()
    split = len(candles) * 2 // 3
    horizon = 24 if interval == "60" else 6

    for index in range(203, len(candles) - horizon):
        if index < 203:
            continue
        rsi_index = index - 14
        atr_index = index - 14
        if rsi_index < 0 or rsi_index >= len(momentum):
            continue
        if atr_index < 0 or atr_index >= len(volatility):
            continue
        current_rsi = momentum[rsi_index]
        touched, direction = touch_confirmed(candles, index)
        if not touched or direction is None:
            continue
        if direction == "LONG":
            if not 50 <= current_rsi <= 65:
                continue
            if not bullish_confirmation(candles, index):
                continue
        else:
            if not 35 <= current_rsi <= 50:
                continue
            if not bearish_confirmation(candles, index):
                continue

        average_volume = sum(
            candle.volume for candle in candles[index - 20:index]
        ) / 20
        if average_volume <= 0 or candles[index].volume < average_volume * variant.volume_multiplier:
            continue

        current_atr = volatility[atr_index]
        entry = candles[index].close
        risk = 1.5 * current_atr
        if risk <= 0 or risk >= entry:
            continue
        result = evaluate(
            direction,
            entry,
            risk,
            candles[index + 1:index + 1 + horizon],
            horizon,
        )
        if result == "pending":
            continue
        counts = train if index < split else validation
        if result == "tp1":
            counts.tp1 += 1
        elif result == "stop":
            counts.stop += 1
        else:
            counts.unresolved += 1
    return train, validation


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", choices=("60", "240"), default="240")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=30) as session:
        client = BybitClient(session)
        symbols = await client.get_top_usdt_perpetual_symbols_by_turnover(args.top)
        now = datetime.now(timezone.utc)
        end = int(now.timestamp() * 1000)
        start = int((now - timedelta(days=args.days)).timestamp() * 1000)
        warmup = 203 * interval_ms(args.interval)
        base_start = start - warmup

        totals = {
            variant.name: (Counts(), Counts()) for variant in VARIANTS
        }
        for symbol in symbols:
            try:
                candles = await fetch_history(
                    client, symbol, args.interval, base_start, end, end
                )
                if not candles:
                    continue
                for variant in VARIANTS:
                    train, validation = scan_variant(candles, variant, args.interval)
                    totals[variant.name][0].tp1 += train.tp1
                    totals[variant.name][0].stop += train.stop
                    totals[variant.name][0].unresolved += train.unresolved
                    totals[variant.name][1].tp1 += validation.tp1
                    totals[variant.name][1].stop += validation.stop
                    totals[variant.name][1].unresolved += validation.unresolved
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue

    print(f"Интервал: {args.interval}м, рынков: {len(symbols)}, история: {args.days} дней")
    for name, (train, validation) in totals.items():
        print(
            f"{name}: train TP1={train.win_rate:.1f}% ({train.resolved}), "
            f"validation TP1={validation.win_rate:.1f}% ({validation.resolved})"
        )


if __name__ == "__main__":
    asyncio.run(main())
