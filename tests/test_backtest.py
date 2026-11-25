import unittest

from backtest import (
    CURRENT_FILTER_NAME,
    OutcomeCounts,
    ema_ribbon_candidate_passes_validation,
    evaluate_outcome,
    interval_milliseconds,
    select_filter_variant,
)
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

    def test_filter_selection_requires_a_holdout_improvement(self):
        candidate = "EMA300 + volume 1.5x"
        training = {
            CURRENT_FILTER_NAME: OutcomeCounts(tp1_first=105, stop_first=95),
            candidate: OutcomeCounts(tp1_first=130, stop_first=70),
        }
        validation = {
            CURRENT_FILTER_NAME: OutcomeCounts(tp1_first=103, stop_first=97),
            candidate: OutcomeCounts(tp1_first=112, stop_first=88),
        }

        self.assertEqual(select_filter_variant(training, validation), candidate)

        validation[candidate] = OutcomeCounts(tp1_first=106, stop_first=94)
        self.assertEqual(select_filter_variant(training, validation), CURRENT_FILTER_NAME)

    def test_filter_selection_returns_none_without_enough_baseline_data(self):
        counts = {
            CURRENT_FILTER_NAME: OutcomeCounts(tp1_first=90, stop_first=90),
            "EMA300 + volume 1.5x": OutcomeCounts(tp1_first=100, stop_first=80),
        }

        self.assertIsNone(select_filter_variant(counts, counts))

    def test_ema_ribbon_pullback_requires_training_and_holdout_improvement(self):
        baseline_train = OutcomeCounts(tp1_first=105, stop_first=95)
        baseline_validation = OutcomeCounts(tp1_first=105, stop_first=95)
        candidate_train = OutcomeCounts(tp1_first=120, stop_first=80)
        candidate_validation = OutcomeCounts(tp1_first=110, stop_first=90)

        self.assertTrue(
            ema_ribbon_candidate_passes_validation(
                baseline_train,
                baseline_validation,
                candidate_train,
                candidate_validation,
            )
        )
        candidate_validation = OutcomeCounts(tp1_first=107, stop_first=93)
        self.assertFalse(
            ema_ribbon_candidate_passes_validation(
                baseline_train,
                baseline_validation,
                candidate_train,
                candidate_validation,
            )
        )


if __name__ == "__main__":
    unittest.main()
