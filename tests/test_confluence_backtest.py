import unittest

from confluence_backtest import (
    Variant,
    bearish_confirmation,
    bullish_confirmation,
    evaluate,
)
from strategy import Candle


class ConfluenceBacktestTests(unittest.TestCase):
    def test_long_and_short_confirmations_are_directional(self):
        bullish = [
            Candle(1, 101, 102, 100, 100),
            Candle(2, 100, 104, 99, 103),
        ]
        bearish = [
            Candle(1, 102, 103, 101, 103),
            Candle(2, 103, 104, 99, 100),
        ]
        self.assertTrue(bullish_confirmation(bullish, 1))
        self.assertTrue(bearish_confirmation(bearish, 1))
        self.assertFalse(bullish_confirmation(bearish, 1))
        self.assertFalse(bearish_confirmation(bullish, 1))

    def test_conservative_stop_hits_before_target(self):
        long_stop = Candle(1, 100, 101, 94, 96)
        long_target = Candle(2, 100, 106, 99, 104)
        self.assertEqual(
            evaluate("LONG", 100, 5, [long_stop, long_target], 24),
            "stop",
        )

    def test_variant_names_distinguish_volume_strictness(self):
        loose = Variant("confluence volume 1.2x", 1.2)
        strict = Variant("confluence volume 2.0x", 2.0)
        self.assertEqual(loose.volume_multiplier, 1.2)
        self.assertEqual(strict.volume_multiplier, 2.0)


if __name__ == "__main__":
    unittest.main()
