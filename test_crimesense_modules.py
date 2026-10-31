import unittest
from datetime import datetime
import numpy as np
import pandas as pd

import temporal_analysis as ta
import grid_risk as gr
import alert_engine as ae
import crime_statistics as cs
import report_validator as rv


def sample_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="7h"),
        "latitude": 41.8 + rng.normal(0, 0.02, n),
        "longitude": -87.65 + rng.normal(0, 0.02, n),
        "primary_type": rng.choice(["THEFT", "BATTERY", "ASSAULT", "ROBBERY"], n),
        "arrest": rng.random(n) < 0.12,
    })


class TemporalTests(unittest.TestCase):
    def test_counts_cover_all_periods(self):
        df = ta.add_temporal_features(sample_df())
        self.assertEqual(len(ta.crimes_by_hour(df)), 24)
        self.assertEqual(len(ta.crimes_by_month(df)), 12)
        self.assertEqual(ta.crimes_by_weekday(df).sum(), len(df))

    def test_bad_dates_dropped(self):
        df = pd.DataFrame({"date": ["2023-01-01 10:00", "garbage"]})
        self.assertEqual(len(ta.add_temporal_features(df)), 1)

    def test_night_flag(self):
        df = ta.add_temporal_features(pd.DataFrame({"date": ["2023-01-01 23:00", "2023-01-01 12:00"]}))
        self.assertEqual(list(df["is_night"]), [1, 0])


class GridTests(unittest.TestCase):
    def test_pipeline_and_no_leakage(self):
        df = ta.add_temporal_features(sample_df(3000))
        df, dlat, dlon = gr.assign_grid(df, 0.5)
        cells = gr.build_cell_features(df, dlat, dlon)
        cells["risk"] = gr.label_risk_terciles(cells["incident_count"])
        self.assertTrue(set(cells["risk"]) <= set(gr.LABELS))
        model, m = gr.train_risk_classifier(cells)
        self.assertIn("accuracy", m)
        with self.assertRaises(ValueError):
            gr.train_risk_classifier(cells, features=["incident_count", "night_fraction"])


class AlertTests(unittest.TestCase):
    def test_haversine_known_distance(self):
        # one degree of latitude ~ 111.2 km
        self.assertAlmostEqual(ae.haversine_km(0, 0, 1, 0), 111.2, delta=0.5)
        self.assertEqual(ae.haversine_km(10, 10, 10, 10), 0.0)

    def test_alert_radius(self):
        zones = [(41.8800, -87.6300)]
        near = (41.8850, -87.6300)   # ~0.55 km
        far = (41.9500, -87.6300)    # ~7.8 km
        self.assertEqual(ae.check_alert(near, zones, 1.0)["alert"], 1)
        self.assertEqual(ae.check_alert(near, zones, 0.3)["alert"], 0)
        self.assertEqual(ae.check_alert(far, zones, 1.0)["alert"], 0)

    def test_coverage_monotonic(self):
        zones = [(41.88, -87.63)]
        users = [(41.88 + i * 0.005, -87.63) for i in range(20)]
        r = [ae.coverage_rate(users, zones, k) for k in (0.5, 1, 2)]
        self.assertTrue(r[0] <= r[1] <= r[2])

    def test_bad_radius(self):
        with self.assertRaises(ValueError):
            ae.check_alert((0, 0), [(0, 0)], 0)


class StatsTests(unittest.TestCase):
    def test_top_types_and_summary(self):
        df = sample_df()
        top = cs.top_crime_types(df, 2)
        self.assertEqual(len(top), 2)
        self.assertEqual(cs.dataset_summary(df)["records"], len(df))
        self.assertTrue(0 <= cs.arrest_rate(df) <= 1)


class ValidatorTests(unittest.TestCase):
    NOW = datetime(2026, 1, 1)

    def good(self):
        return {"latitude": 12.9, "longitude": 79.1, "description": "Chain snatching near bus stand",
                "category": "theft", "date": "2025-12-30"}

    def test_valid(self):
        ok, errs = rv.validate_report(self.good(), self.NOW)
        self.assertTrue(ok, errs)

    def test_invalid_cases(self):
        r = self.good(); r["latitude"] = 200
        self.assertFalse(rv.validate_report(r, self.NOW)[0])
        r = self.good(); r["description"] = "Lovely weather today"; r["category"] = "other"
        self.assertFalse(rv.validate_report(r, self.NOW)[0])
        r = self.good(); r["date"] = "2027-01-01"
        self.assertFalse(rv.validate_report(r, self.NOW)[0])

    def test_boundary_matching(self):
        self.assertFalse(rv.has_crime_keyword("thefts and robberies"))  # whole words only
        self.assertTrue(rv.has_crime_keyword("There was a THEFT."))


if __name__ == "__main__":
    unittest.main()
