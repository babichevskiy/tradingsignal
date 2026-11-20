import unittest
from unittest.mock import patch

from strategy import Candle, atr, build_signal, detect_rsi_extreme, ema, rsi


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
        candles = [Candle(index, 100, 101, 99, 100) for index in range(60)]
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
        candles = [Candle(index, 100, 101, 99, 100) for index in range(60)]
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

    def test_no_signal_when_crossover_is_not_confirmed(self):
        candles = [
            Candle(index, 100, 100.2, 99.8, 100)
            for index in range(60)
        ]
        self.assertIsNone(build_signal("TESTUSDT", candles))


if __name__ == "__main__":
    unittest.main()
