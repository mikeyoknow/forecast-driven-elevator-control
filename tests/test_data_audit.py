from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.audit import day_manifest, minute_demand_frame
from elevator_ml.data.generator import create_days


class DataAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.days = create_days(
            split="train",
            scenarios=("morning", "lunch"),
            days_per_scenario=2,
            seed_offset=10,
            duration_minutes=6,
            n_floors=5,
        )

    def test_day_manifest_matches_generated_days(self) -> None:
        manifest = day_manifest(self.days)

        self.assertEqual(len(manifest), 4)
        self.assertEqual(manifest["seed"].nunique(), 4)
        self.assertEqual(
            int(manifest["passengers"].sum()),
            sum(len(day.passengers) for day in self.days),
        )

    def test_minute_frame_has_one_row_per_day_minute(self) -> None:
        frame = minute_demand_frame(self.days)

        self.assertEqual(len(frame), 4 * 6)
        self.assertEqual(
            int(frame["total_arrivals"].sum()),
            sum(len(day.passengers) for day in self.days),
        )


if __name__ == "__main__":
    unittest.main()
