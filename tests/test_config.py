from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.config import BuildingConfig, ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_development_config_loads(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "development.json")

        self.assertEqual(config.experiment_name, "development_v1")
        self.assertEqual(config.building.n_floors, 15)
        self.assertEqual(config.traffic.train_scenarios[0], "morning")
        self.assertEqual(config.forecast.history_windows_minutes, (1, 3, 5, 10))
        self.assertEqual(config.forecast.target_horizons_minutes, (1, 2, 5))
        self.assertEqual(config.forecast.primary_horizon_minutes, 2)
        self.assertEqual(config.models.ridge_alphas, (1.0, 10.0, 50.0))
        self.assertEqual(config.models.random_state, 3404)
        self.assertEqual(config.evaluation.positioning_intervals_minutes, (1, 2, 5))
        self.assertEqual(config.evaluation.high_demand_multiplier, 1.5)
        self.assertEqual(config.sequence_models.sequence_lengths, (5, 10))
        self.assertEqual(config.sequence_models.hidden_sizes, (16, 32))
        self.assertEqual(config.sequence_models.gradient_clip_norm, 5.0)

    def test_development_config_does_not_contain_final_holdout_seed(self) -> None:
        raw = json.loads(
            (PROJECT_ROOT / "configs" / "development.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertNotIn("final", raw["seeds"])
        self.assertNotIn("final_test", raw["seeds"])

    def test_invalid_building_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            BuildingConfig(
                n_floors=1,
                n_elevators=4,
                capacity=12,
                seconds_per_floor=2,
                door_seconds=4,
                maximum_drain_minutes=30,
            )

    def test_unknown_top_level_key_is_rejected(self) -> None:
        raw = json.loads(
            (PROJECT_ROOT / "configs" / "development.json").read_text(
                encoding="utf-8"
            )
        )
        raw["surprise"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
