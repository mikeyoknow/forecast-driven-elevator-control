from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.config import load_config
from elevator_ml.evaluation.final_holdout import (
    create_final_days,
    validate_final_protocol,
)


class FinalProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "development.json")
        self.protocol = json.loads(
            (PROJECT_ROOT / "configs" / "final_protocol.json").read_text(
                encoding="utf-8"
            )
        )

    def test_unlocked_protocol_is_rejected(self) -> None:
        self.protocol["status"] = "pre_seed_lock"
        with self.assertRaises(ValueError):
            validate_final_protocol(self.protocol, self.config)

    def test_locked_protocol_and_seed_ranges(self) -> None:
        validate_final_protocol(self.protocol, self.config)
        days = create_final_days(self.config, self.protocol, 2_000_000)

        self.assertEqual(len(days["in_distribution"]), 20)
        self.assertEqual(len(days["unseen_surge"]), 8)
        self.assertEqual(len(days["high_demand"]), 8)
        seeds = [day.seed for group in days.values() for day in group]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(day.split.startswith("final_") for group in days.values() for day in group))


if __name__ == "__main__":
    unittest.main()
