"""Unit tests. Run with:  py -m unittest test_detect.py"""
import unittest

from can_bus import FAULTS, KNOWN_IDS, inject, labelled_run, normal_traffic
from detect import ForestDetector, RuleBaseline, evaluate
from features import FEATURE_NAMES, to_matrix, window_rows


class TestSimulator(unittest.TestCase):
    def test_normal_traffic_is_sorted_and_known(self):
        frames = normal_traffic(seconds=5.0, seed=1)
        self.assertGreater(len(frames), 0)
        times = [f[0] for f in frames]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(f[1] in KNOWN_IDS for f in frames))

    def test_same_seed_same_traffic(self):
        self.assertEqual(normal_traffic(seconds=3.0, seed=4),
                         normal_traffic(seconds=3.0, seed=4))

    def test_different_seed_different_traffic(self):
        self.assertNotEqual(normal_traffic(seconds=3.0, seed=4),
                            normal_traffic(seconds=3.0, seed=5))

    def test_periods_are_roughly_right(self):
        frames = normal_traffic(seconds=20.0, seed=2)
        fast = [f[0] for f in frames if f[1] == 0x0A0]
        gaps = [b - a for a, b in zip(fast, fast[1:])]
        mean_ms = 1000 * sum(gaps) / len(gaps)
        self.assertAlmostEqual(mean_ms, 10.0, delta=1.0)

    def test_dos_adds_frames_only_in_the_window(self):
        frames = normal_traffic(seconds=20.0, seed=3)
        out = inject(frames, "dos", 5.0, 2.0, seed=3)
        flood = [f for f in out if f[1] == 0x000]
        self.assertGreater(len(flood), 1000)
        self.assertTrue(all(5.0 <= f[0] < 7.0 for f in flood))

    def test_suspension_removes_the_victim_only_in_the_window(self):
        frames = normal_traffic(seconds=20.0, seed=3)
        out = inject(frames, "suspension", 5.0, 5.0, seed=3)
        during = [f for f in out if f[1] == 0x0B0 and 5.0 <= f[0] < 10.0]
        after = [f for f in out if f[1] == 0x0B0 and f[0] >= 10.0]
        self.assertEqual(during, [])
        self.assertGreater(len(after), 0)

    def test_fuzzing_introduces_unknown_ids(self):
        frames = normal_traffic(seconds=20.0, seed=3)
        out = inject(frames, "fuzzing", 5.0, 5.0, seed=3)
        unknown = [f for f in out if f[1] not in KNOWN_IDS]
        self.assertGreater(len(unknown), 0)

    def test_drift_changes_payload_not_timing(self):
        frames = normal_traffic(seconds=40.0, seed=3)
        out = inject(frames, "drift", 10.0, 20.0, seed=3)
        self.assertEqual(len(frames), len(out))
        before = [f for f in frames if f[1] == 0x0E0 and 10.0 <= f[0] < 30.0]
        after = [f for f in out if f[1] == 0x0E0 and 10.0 <= f[0] < 30.0]
        self.assertNotEqual([f[2] for f in before], [f[2] for f in after])

    def test_unknown_fault_is_rejected(self):
        with self.assertRaises(ValueError):
            inject(normal_traffic(seconds=2.0), "sabotage", 0.5, 0.5)

    def test_labelled_run_returns_the_window(self):
        _, window = labelled_run("dos", seconds=30.0, start_s=10.0, duration_s=5.0)
        self.assertEqual(window, (10.0, 15.0))


class TestFeatures(unittest.TestCase):
    def test_empty_input(self):
        rows, times = window_rows([], KNOWN_IDS)
        self.assertEqual(rows, [])
        self.assertEqual(times, [])

    def test_every_row_has_every_feature(self):
        frames = normal_traffic(seconds=5.0, seed=6)
        rows, times = window_rows(frames, KNOWN_IDS)
        self.assertEqual(len(rows), len(times))
        for row in rows:
            self.assertEqual(set(row), set(FEATURE_NAMES))
            for value in row.values():
                self.assertIsInstance(value, float)

    def test_matrix_column_order_is_fixed(self):
        frames = normal_traffic(seconds=3.0, seed=6)
        rows, _ = window_rows(frames, KNOWN_IDS)
        matrix = to_matrix(rows)
        self.assertEqual(len(matrix[0]), len(FEATURE_NAMES))
        self.assertEqual(matrix[0][0], rows[0]["frames_per_s"])

    def test_dos_raises_the_frame_rate(self):
        frames = normal_traffic(seconds=20.0, seed=7)
        clean, _ = window_rows(frames, KNOWN_IDS)
        attacked, times = window_rows(
            inject(frames, "dos", 5.0, 5.0, seed=7), KNOWN_IDS)
        idx = [i for i, t in enumerate(times) if 5.0 <= t < 10.0]
        self.assertGreater(max(attacked[i]["frames_per_s"] for i in idx),
                           5 * max(r["frames_per_s"] for r in clean))

    def test_fuzzing_raises_unknown_id_count(self):
        frames = normal_traffic(seconds=20.0, seed=7)
        rows, times = window_rows(
            inject(frames, "fuzzing", 5.0, 5.0, seed=7), KNOWN_IDS)
        during = [r["unknown_id_frames"] for r, t in zip(rows, times)
                  if 5.0 <= t < 10.0]
        outside = [r["unknown_id_frames"] for r, t in zip(rows, times)
                   if not (5.0 <= t < 10.0)]
        self.assertGreater(sum(during), 0)
        self.assertEqual(sum(outside), 0)


class TestDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = normal_traffic(seconds=60.0, seed=10)
        cls.calib = normal_traffic(seconds=60.0, seed=11)
        cls.train_rows, _ = window_rows(cls.train, KNOWN_IDS)
        cls.calib_rows, _ = window_rows(cls.calib, KNOWN_IDS)

    def test_rules_do_not_fire_on_healthy_traffic(self):
        rules = RuleBaseline().fit(self.train, self.train_rows)
        pred = rules.predict(self.calib_rows)
        self.assertLess(pred.mean(), 0.05)

    def test_rules_catch_a_flood(self):
        rules = RuleBaseline().fit(self.train, self.train_rows)
        frames, window = labelled_run("dos", seconds=60.0, start_s=20.0,
                                      duration_s=10.0, seed=12)
        rows, times = window_rows(frames, KNOWN_IDS)
        result = evaluate(rules.predict(rows), times, 0.2, window)
        self.assertGreater(result["detection_rate"], 0.9)

    def test_forest_needs_fit_before_predict(self):
        with self.assertRaises(RuntimeError):
            ForestDetector().predict(self.calib_rows)

    def test_forest_threshold_matches_target_fpr(self):
        forest = ForestDetector(target_fpr=0.05).fit(self.train_rows,
                                                     self.calib_rows)
        rate = forest.predict(self.calib_rows).mean()
        self.assertAlmostEqual(rate, 0.05, delta=0.02)

    def test_forest_catches_a_flood(self):
        forest = ForestDetector().fit(self.train_rows, self.calib_rows)
        frames, window = labelled_run("dos", seconds=60.0, start_s=20.0,
                                      duration_s=10.0, seed=12)
        rows, times = window_rows(frames, KNOWN_IDS)
        result = evaluate(forest.predict(rows), times, 0.2, window)
        self.assertGreater(result["detection_rate"], 0.9)

    def test_forest_beats_rules_on_a_silent_node(self):
        rules = RuleBaseline().fit(self.train, self.train_rows)
        forest = ForestDetector().fit(self.train_rows, self.calib_rows)
        frames, window = labelled_run("suspension", seconds=60.0, start_s=20.0,
                                      duration_s=20.0, seed=12)
        rows, times = window_rows(frames, KNOWN_IDS)
        r = evaluate(rules.predict(rows), times, 0.2, window)
        f = evaluate(forest.predict(rows), times, 0.2, window)
        self.assertGreater(f["detection_rate"], r["detection_rate"])

    def test_hint_names_a_real_feature(self):
        forest = ForestDetector().fit(self.train_rows, self.calib_rows)
        hints = forest.feature_importance_hint(self.calib_rows[:5])
        self.assertTrue(all(h in FEATURE_NAMES for h in hints))

    def test_evaluate_counts_correctly(self):
        pred = [0, 1, 1, 0, 1]
        times = [0.0, 0.2, 0.4, 0.6, 0.8]
        result = evaluate(pred, times, 0.2, (0.2, 0.6))
        self.assertEqual(result["windows_in_fault"], 2)
        self.assertEqual(result["detected"], 2)
        self.assertEqual(result["detection_rate"], 1.0)
        self.assertEqual(result["false_alarms"], 1)

    def test_every_fault_runs_end_to_end(self):
        forest = ForestDetector().fit(self.train_rows, self.calib_rows)
        for fault in FAULTS:
            frames, window = labelled_run(fault, seconds=40.0, start_s=15.0,
                                          duration_s=10.0, seed=13)
            rows, times = window_rows(frames, KNOWN_IDS)
            result = evaluate(forest.predict(rows), times, 0.2, window)
            self.assertIn("detection_rate", result)
            self.assertGreaterEqual(result["detection_rate"], 0.0)
            self.assertLessEqual(result["detection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
