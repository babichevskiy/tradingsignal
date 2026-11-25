import unittest
from unittest.mock import patch

from strategy import (
    Candle,
    Signal,
    atr,
    build_signal,
    build_ema_ribbon_pullback_signals,
    confirm_trend,
    confirm_ema_ribbon,
    detect_rsi_extreme,
    ema,
    rsi,
)


class IndicatorTests(unittest.TestCase):
    def test_ema_requires_enough_values_and_tracks_average(self):
        self.assertEqual(ema([1, 2, 3], 3), [2])
        self.assertEqual(ema([1, 2], 3), [])

    def test_rsi_is_bounded_and_handles_flat_market(self):
        self.assertEqual(rsi([10] * 16)[-1], 50)
        self.assertEqual(rsi(list(range(16)))[-1], 100)

    def test_detects_rsi_overbought_oversold_and_neutral(self):
        rising = list(range(16))
        falling = list(range(16, 0, -1))
        flat = [10] * 16

        self.assertEqual(detect_rsi_extreme(rising), ("overbought", 100))
        self.assertEqual(detect_rsi_extreme(falling), ("oversold", 0))
        self.assertIsNone(detect_rsi_extreme(flat))
        self.assertIsNone(detect_rsi_extreme([1, 2], period=14))

    def test_atr_uses_true_range_across_gaps(self):
        candles = [
            Candle(index, 10, 11, 9, 10)
            for index in range(15)
        ]
        candles.append(Candle(15, 12, 14, 12, 13))
        self.assertAlmostEqual(atr(candles, 14)[-1], 30 / 14)


class SignalTests(unittest.TestCase):
    def test_signal_levels_are_directional_and_risk_based(self):
        candles = [
            Candle(index, 100, 101, 99, 100, 10)
            for index in range(59)
        ] + [Candle(59, 100, 101, 99, 100, 12)]
        with (
            patch("strategy.ema", side_effect=[[1, 2], [2, 1]]),
            patch("strategy.rsi", return_value=[55, 60]),
            patch("strategy.atr", return_value=[1, 2]),
        ):
            signal = build_signal("TESTUSDT", candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.stop_loss, 97)
        self.assertEqual(signal.take_profit_1, 103)
        self.assertEqual(signal.take_profit_2, 106)

    def test_short_signal_levels_are_directional(self):
        candles = [
            Candle(index, 100, 101, 99, 100, 10)
            for index in range(59)
        ] + [Candle(59, 100, 101, 99, 100, 12)]
        with (
            patch("strategy.ema", side_effect=[[2, 1], [1, 2]]),
            patch("strategy.rsi", return_value=[45, 40]),
            patch("strategy.atr", return_value=[1, 2]),
        ):
            signal = build_signal("TESTUSDT", candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "SHORT")
        self.assertEqual(signal.stop_loss, 103)
        self.assertEqual(signal.take_profit_1, 97)
        self.assertEqual(signal.take_profit_2, 94)

    def test_signal_is_rejected_when_volume_confirmation_is_missing(self):
        candles = [Candle(index, 100, 101, 99, 100, 10) for index in range(60)]
        with (
            patch("strategy.ema", side_effect=[[1, 2], [2, 1]]),
            patch("strategy.rsi", return_value=[55, 60]),
            patch("strategy.atr", return_value=[1, 2]),
        ):
            signal = build_signal("TESTUSDT", candles, volume_multiplier=1.2)

        self.assertIsNone(signal)

    def test_trend_filter_accepts_only_signals_aligned_with_ema(self):
        long_signal = Signal("TESTUSDT", "LONG", 110, 113, 116, 107, 0)
        short_signal = Signal("TESTUSDT", "SHORT", 90, 87, 84, 93, 0)
        uptrend = [Candle(index, 100, 101, 99, 100) for index in range(200)]
        uptrend.append(Candle(200, 110, 111, 109, 110))
        downtrend = [Candle(index, 100, 101, 99, 100) for index in range(200)]
        downtrend.append(Candle(200, 90, 91, 89, 90))

        self.assertTrue(confirm_trend(long_signal, uptrend))
        self.assertFalse(confirm_trend(short_signal, uptrend))
        self.assertTrue(confirm_trend(short_signal, downtrend))
        self.assertFalse(confirm_trend(long_signal, downtrend))
        self.assertFalse(confirm_trend(long_signal, uptrend[:100]))

    def test_ema_ribbon_filter_requires_ordered_levels_and_price_position(self):
        long_signal = Signal("TESTUSDT", "LONG", 110, 113, 116, 107, 0)
        candles = [Candle(index, 100, 111, 99, 110) for index in range(201)]

        with patch("strategy.ema", side_effect=[[108] * 182, [106] * 152, [104] * 102, [102, 102]]):
            self.assertTrue(confirm_ema_ribbon(long_signal, candles))
        with patch("strategy.ema", side_effect=[[108] * 182, [109] * 152, [104] * 102, [102, 102]]):
            self.assertFalse(confirm_ema_ribbon(long_signal, candles))

    def test_ema_ribbon_pullback_generates_signal_after_reclaim_with_volume(self):
        candles = [Candle(index, 100, 101, 99, 100, 10) for index in range(201)]
        candles[-2] = Candle(199, 100, 101, 99, 104, 10)
        candles[-1] = Candle(200, 104, 107, 103, 106, 12)

        def fake_ema(values, period):
            level = {20: 105, 50: 104, 100: 103, 200: 102}[period]
            return [level] * (len(values) - period + 1)

        with (
            patch("strategy.ema", side_effect=fake_ema),
            patch("strategy.rsi", return_value=[60] * (len(candles) - 14)),
            patch("strategy.atr", return_value=[1] * (len(candles) - 14)),
        ):
            signals = build_ema_ribbon_pullback_signals("TESTUSDT", candles)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "TESTUSDT")
        self.assertEqual(signals[0].direction, "LONG")
        self.assertEqual(signals[0].entry, 106)
        self.assertEqual(signals[0].stop_loss, 104.5)
        self.assertEqual(signals[0].take_profit_1, 107.5)

    def test_no_signal_when_crossover_is_not_confirmed(self):
        candles = [
            Candle(index, 100, 100.2, 99.8, 100)
            for index in range(60)
        ]
        self.assertIsNone(build_signal("TESTUSDT", candles))


if __name__ == "__main__":
    unittest.main()
