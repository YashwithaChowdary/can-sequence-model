"""Tests for the sequence detector.

Split in two: the data-shaping tests run in milliseconds, and the handful that
need a trained model share one model trained once in setUpClass. The trained
tests assert the direction of the result rather than exact numbers, because
the numbers move a little with the seed, but the claim the project is built on
-- that this model sees drift and replay where window features do not -- has
to hold or the project has failed.

    py -m unittest test_sequence.py
"""
import unittest

import numpy as np

from can_bus import KNOWN_IDS, labelled_run, normal_traffic
from detect import ForestDetector, evaluate
from features import WINDOW_S, window_rows
from sequence_data import (
    BYTE_SCALE, HISTORY, MAX_DLC, build_sequences, n_windows_for,
    per_id_error_stats, window_scores,
)
from sequence_model import SequenceDetector


class TestSequenceData(unittest.TestCase):

    def test_unknown_ids_are_skipped(self):
        frames = [(i * 0.01, 0x7FF, [1, 2, 3]) for i in range(40)]
        data = build_sequences(frames, KNOWN_IDS)
        self.assertEqual(len(data["time"]), 0,
                         "an id with no embedding cannot be scored at all")

    def test_first_frames_of_each_id_have_no_history(self):
        frames = [(i * 0.01, 0x0A0, [i % 256] * 8) for i in range(HISTORY + 3)]
        data = build_sequences(frames, KNOWN_IDS)
        self.assertEqual(len(data["time"]), 3,
                         "the first HISTORY frames are context, not samples")

    def test_history_is_oldest_first_and_matches_previous_payloads(self):
        frames = [(i * 0.01, 0x0A0, [i] * 8) for i in range(HISTORY + 1)]
        data = build_sequences(frames, KNOWN_IDS)
        self.assertEqual(data["history"].shape, (1, HISTORY, MAX_DLC))
        # the single sample's history should be frames 0..HISTORY-1 in order
        expected = np.array([[i / BYTE_SCALE] * MAX_DLC for i in range(HISTORY)],
                            dtype=np.float32)
        np.testing.assert_allclose(data["history"][0], expected, atol=1e-6)
        # and its target should be the last frame
        np.testing.assert_allclose(
            data["target"][0], np.full(MAX_DLC, HISTORY / BYTE_SCALE), atol=1e-6)

    def test_short_payloads_are_zero_padded(self):
        frames = [(i * 0.01, 0x0D1, [0x40, 0x41]) for i in range(HISTORY + 1)]
        data = build_sequences(frames, KNOWN_IDS)
        target = data["target"][0]
        self.assertAlmostEqual(float(target[0]), 0x40 / BYTE_SCALE, places=5)
        self.assertEqual(list(target[2:]), [0.0] * (MAX_DLC - 2))

    def test_window_count_matches_the_feature_rows(self):
        frames = normal_traffic(seconds=12.0, seed=3)
        rows, times = window_rows(frames, KNOWN_IDS, WINDOW_S)
        self.assertEqual(n_windows_for(frames, WINDOW_S), len(rows),
                         "sequence scores must line up with feature rows")


class TestPerIdStats(unittest.TestCase):

    def test_a_perfectly_predicted_id_gets_the_floor_not_zero(self):
        errors = np.zeros(50)
        id_index = np.zeros(50, dtype=np.int64)
        centre, scale = per_id_error_stats(errors, id_index, 1, floor=0.05)
        self.assertEqual(centre[0], 0.0)
        self.assertEqual(scale[0], 0.05, "scale must never be zero")

    def test_scale_reflects_each_ids_own_spread(self):
        # id 0 is easy and consistent, id 1 is noisy
        errors = np.concatenate([np.full(100, 0.5), np.linspace(0, 30, 100)])
        id_index = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int64)
        centre, scale = per_id_error_stats(errors, id_index, 2)
        self.assertLess(scale[0], scale[1],
                        "a noisy id must be judged on a wider scale")


class TestWindowScores(unittest.TestCase):

    def test_windows_without_frames_score_zero(self):
        scores = window_scores(np.array([5.0]), np.array([0.1]),
                               np.array([0], dtype=np.int64), 4, 0.2)
        self.assertEqual(scores[0], 5.0)
        self.assertEqual(list(scores[1:]), [0.0, 0.0, 0.0])

    def test_worst_id_sets_the_window_score(self):
        # two ids in one window: one normal, one far out
        errors = np.array([0.1, 0.1, 9.0, 9.0])
        times = np.array([0.01, 0.02, 0.03, 0.04])
        ids = np.array([0, 0, 1, 1], dtype=np.int64)
        scores = window_scores(errors, times, ids, 1, 0.2)
        self.assertAlmostEqual(scores[0], 9.0,
                               msg="the worst id, not the average, sets the score")

    def test_standardisation_is_relative_to_each_id(self):
        # id 1 normally errs by 10; an error of 10 is unremarkable for it
        errors = np.array([10.0, 10.0])
        times = np.array([0.01, 0.02])
        ids = np.array([0, 1], dtype=np.int64)
        centre = np.array([0.0, 10.0])
        scale = np.array([1.0, 1.0])
        scores = window_scores(errors, times, ids, 1, 0.2,
                               centre=centre, scale=scale)
        # id 0 is 10 scales out, id 1 is 0 scales out -> id 0 should dominate
        self.assertAlmostEqual(scores[0], 10.0)


class TestTrainedDetector(unittest.TestCase):
    """One model, trained once, shared by the tests that need it."""

    @classmethod
    def setUpClass(cls):
        cls.train = normal_traffic(seconds=60.0, seed=0)
        cls.calib = normal_traffic(seconds=60.0, seed=101)
        cls.seq = SequenceDetector(
            KNOWN_IDS, target_fpr=0.02, epochs=12, seed=0,
        ).fit(cls.train, cls.calib, verbose=False)

    def test_training_reduced_the_error(self):
        losses = self.seq.train_loss
        self.assertLess(losses[-1], losses[0] / 2,
                        "the model should learn something in 12 epochs")

    def test_threshold_matches_the_requested_false_positive_rate(self):
        held_out = normal_traffic(seconds=60.0, seed=202)
        pred = self.seq.predict(held_out)
        rate = pred.mean()
        self.assertLess(rate, 0.08,
                        f"false positive rate {rate:.1%} on unseen healthy traffic")

    def test_unknown_traffic_produces_no_scores(self):
        frames = [(i * 0.01, 0x7FE, [i % 256] * 8) for i in range(200)]
        errors, times, ids = self.seq.frame_errors(frames)
        self.assertEqual(len(errors), 0)

    def test_beats_the_forest_on_drift(self):
        """The central claim of the project."""
        frames, fault_window = labelled_run("drift", seconds=60.0,
                                            start_s=24.0, duration_s=12.0, seed=7)
        rows, times = window_rows(frames, KNOWN_IDS, WINDOW_S)
        train_rows, _ = window_rows(self.train, KNOWN_IDS, WINDOW_S)
        calib_rows, _ = window_rows(self.calib, KNOWN_IDS, WINDOW_S)
        forest = ForestDetector(target_fpr=0.02).fit(train_rows, calib_rows)

        s_pred = self.seq.predict(frames)
        f_pred = forest.predict(rows)
        n = min(len(s_pred), len(f_pred), len(times))
        s_rate = evaluate(s_pred[:n], times[:n], WINDOW_S, fault_window)["detection_rate"]
        f_rate = evaluate(f_pred[:n], times[:n], WINDOW_S, fault_window)["detection_rate"]
        self.assertGreater(s_rate, f_rate + 0.3,
                           f"sequence {s_rate:.0%} vs forest {f_rate:.0%} on drift")

    def test_beats_the_forest_on_replay(self):
        frames, fault_window = labelled_run("replay", seconds=60.0,
                                            start_s=24.0, duration_s=12.0, seed=7)
        rows, times = window_rows(frames, KNOWN_IDS, WINDOW_S)
        train_rows, _ = window_rows(self.train, KNOWN_IDS, WINDOW_S)
        calib_rows, _ = window_rows(self.calib, KNOWN_IDS, WINDOW_S)
        forest = ForestDetector(target_fpr=0.02).fit(train_rows, calib_rows)

        s_pred = self.seq.predict(frames)
        f_pred = forest.predict(rows)
        n = min(len(s_pred), len(f_pred), len(times))
        s_rate = evaluate(s_pred[:n], times[:n], WINDOW_S, fault_window)["detection_rate"]
        f_rate = evaluate(f_pred[:n], times[:n], WINDOW_S, fault_window)["detection_rate"]
        self.assertGreater(s_rate, f_rate + 0.3,
                           f"sequence {s_rate:.0%} vs forest {f_rate:.0%} on replay")

    def test_blames_the_message_that_actually_drifted(self):
        frames, fault_window = labelled_run("drift", seconds=60.0,
                                            start_s=24.0, duration_s=12.0, seed=7)
        pred = self.seq.predict(frames)
        blamed = self.seq.worst_id_per_window(frames)
        start, end = fault_window
        hits = [blamed[i] for i in range(min(len(pred), len(blamed)))
                if pred[i] and start <= i * WINDOW_S < end and blamed[i] is not None]
        self.assertTrue(hits, "nothing was flagged inside the fault")
        top = max(set(hits), key=hits.count)
        self.assertEqual(top, 0x0E0,
                         "drift is injected into 0x0E0; the model should say so")

    def test_misses_a_flood_and_says_nothing_rather_than_guessing(self):
        """An honest negative: the flood uses an id with no embedding."""
        frames, fault_window = labelled_run("dos", seconds=60.0,
                                            start_s=24.0, duration_s=12.0, seed=7)
        pred = self.seq.predict(frames)
        times = [i * WINDOW_S for i in range(len(pred))]
        rate = evaluate(pred, times, WINDOW_S, fault_window)["detection_rate"]
        self.assertLess(rate, 0.25,
                        "this detector is not supposed to catch a flood; "
                        "the rule baseline is")


if __name__ == "__main__":
    unittest.main(verbosity=2)
