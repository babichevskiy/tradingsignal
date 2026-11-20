import unittest

from backtest import OutcomeCounts, evaluate_outcome, interval_milliseconds
from strategy import Candle, Signal


class BacktestTests(unittest.TestCase):
    def test_interval_milliseconds(self):
        self.assertEqual(interval_milliseconds("15"), 900_000)
        self.assertEqual(interval_milliseconds("60"), 3_600_000)
        self.assertEqual(interval_milliseconds("D"), 86_400_000)

    def test_long_target_and_stop_outcomes(self):
        long_signal = Signal("BTCUSDT", "LONG", 100, 105, 110, 95, 0)
        target = Candle(1, 100, 106, 99, 104)
        stop = Candle(2, 100, 101, 94, 96)

        self.assertEqual(evaluate_outcome(long_signal, [target], 24), "tp1")
        self.assertEqual(evaluate_outcome(long_signal, [stop], 24), "stop")

    def test_stop_wins_if_target_and_stop_touch_in_same_candle(self):
        signal = Signal("BTCUSDT", "LONG", 100, 105, 110, 95, 0)
        ambiguous = Candle(1, 100, 106, 94, 100)

        self.assertEqual(evaluate_outcome(signal, [ambiguous], 24), "stop")

    def test_counts_separate_resolved_pending_and_unresolved_signals(self):
        counts = OutcomeCounts()
        for result in ("tp1", "stop", "unresolved", "pending"):
            counts.add(result)

        self.assertEqual(
            (
                counts.signals,
                counts.tp1_first,
                counts.stop_first,
                counts.unresolved,
                counts.pending,
            ),
            (4, 1, 1, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
