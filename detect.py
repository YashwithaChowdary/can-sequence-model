"""Two detectors and one honest way of choosing their thresholds.

Both are trained on healthy traffic only, because that is the situation on a
real vehicle: normal traffic is plentiful and faults are not. Neither detector
sees a labelled fault during training.

The rule baseline exists so that the machine learning model has something to
beat. A model that cannot beat three rules is not worth running on an embedded
board.
"""
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, to_matrix


class RuleBaseline:
    """Three rules learned from healthy traffic.

    1. an id that was never seen while healthy is suspicious
    2. a frame rate far above anything seen while healthy is suspicious
    3. a message that should be periodic going missing is suspicious
    """

    def __init__(self, margin=1.5):
        self.margin = margin
        self.known_ids = set()
        self.max_rate = 0.0
        self.expected_ids = 0.0

    def fit(self, frames, rows):
        self.known_ids = {f[1] for f in frames}
        rates = [r["frames_per_s"] for r in rows]
        self.max_rate = max(rates) if rates else 0.0
        seen = [r["known_ids_seen"] for r in rows]
        # the 10th percentile, so a normally quiet window is not an alarm
        self.expected_ids = float(np.percentile(seen, 10)) if seen else 0.0
        return self

    def predict(self, rows):
        """1 for anomalous, 0 for normal."""
        out = []
        for r in rows:
            unknown = r["unknown_id_frames"] > 0
            flooding = r["frames_per_s"] > self.max_rate * self.margin
            silent = r["known_ids_seen"] < self.expected_ids - 1
            out.append(1 if (unknown or flooding or silent) else 0)
        return np.array(out)


class ForestDetector:
    """Isolation Forest on window features, trained on healthy traffic only.

    The threshold is chosen on a held-out slice of healthy traffic, at a
    target false positive rate. It is never chosen on data containing faults,
    because a threshold tuned that way reports a score the deployed system
    will not reproduce.
    """

    def __init__(self, target_fpr=0.02, n_estimators=100, random_state=0):
        self.target_fpr = target_fpr
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination="auto",
            random_state=random_state,
        )
        self.threshold = None

    def fit(self, train_rows, calib_rows):
        x = self.scaler.fit_transform(np.array(to_matrix(train_rows)))
        self.model.fit(x)
        calib = self.scaler.transform(np.array(to_matrix(calib_rows)))
        scores = self.model.score_samples(calib)   # higher is more normal
        # flag the lowest target_fpr of healthy windows
        self.threshold = float(np.quantile(scores, self.target_fpr))
        return self

    def scores(self, rows):
        x = self.scaler.transform(np.array(to_matrix(rows)))
        return self.model.score_samples(x)

    def predict(self, rows):
        if self.threshold is None:
            raise RuntimeError("fit() must be called before predict()")
        return (self.scores(rows) < self.threshold).astype(int)

    def feature_importance_hint(self, rows):
        """Which feature is furthest from the training mean, per flagged window.

        Not a true attribution, but enough to point an engineer at a starting
        place rather than handing them a bare alarm.
        """
        x = self.scaler.transform(np.array(to_matrix(rows)))
        hints = []
        for row in x:
            idx = int(np.argmax(np.abs(row)))
            hints.append(FEATURE_NAMES[idx])
        return hints


def evaluate(pred, times, window_s, fault_window):
    """Detection rate inside the faulty window, false alarms outside it."""
    start, end = fault_window
    inside = [i for i, t in enumerate(times) if start <= t < end]
    outside = [i for i, t in enumerate(times) if not (start <= t < end)]
    detected = sum(pred[i] for i in inside)
    false_alarms = sum(pred[i] for i in outside)
    return {
        "windows_in_fault": len(inside),
        "detected": int(detected),
        "detection_rate": detected / len(inside) if inside else 0.0,
        "false_alarms": int(false_alarms),
        "false_alarm_rate": false_alarms / len(outside) if outside else 0.0,
    }
