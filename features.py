"""Turn a list of CAN frames into one row of numbers per time window.

The detector never sees raw frames. It sees windows, because a single frame
is rarely anomalous on its own: what gives a fault away is the rate, the
timing regularity or the payload behaviour over a short stretch of time.

Window length is a trade-off. Short windows react quickly but are noisy for
slow messages; long windows are stable but hide short bursts. 200 ms is used
by default, which is two periods of the slowest message that matters.
"""
import math
from collections import defaultdict

WINDOW_S = 0.2


def _entropy(values):
    """Shannon entropy in bits of a list of byte values, 0 when all the same."""
    if not values:
        return 0.0
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def window_rows(frames, known_ids, window_s=WINDOW_S):
    """Split frames into fixed windows and describe each one.

    Returns (rows, times) where rows is a list of dicts of plain numbers and
    times is the start time of each window.
    """
    if not frames:
        return [], []
    end = frames[-1][0]
    n_windows = int(end / window_s) + 1
    buckets = [[] for _ in range(n_windows)]
    for t, can_id, data in frames:
        idx = int(t / window_s)
        if idx < n_windows:
            buckets[idx].append((t, can_id, data))

    known = set(known_ids)
    rows, times = [], []
    for i, bucket in enumerate(buckets):
        rows.append(_describe(bucket, known, window_s))
        times.append(i * window_s)
    return rows, times


def _describe(bucket, known, window_s):
    """One window of frames as a flat dict of features."""
    n = len(bucket)
    ids = [f[1] for f in bucket]
    unique_ids = set(ids)
    unknown = [i for i in ids if i not in known]

    # inter-arrival times per id, pooled: regular traffic has small spread
    by_id = defaultdict(list)
    for t, can_id, _ in bucket:
        by_id[can_id].append(t)
    gaps = []
    for times in by_id.values():
        times.sort()
        gaps.extend(b - a for a, b in zip(times, times[1:]))
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    var_gap = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) if gaps else 0.0

    # payload behaviour: entropy of all bytes, and how much bytes move frame
    # to frame within the same id. A replayed or frozen signal moves too little;
    # a fuzzer moves far too much.
    all_bytes = [b for _, _, data in bucket for b in data]
    deltas = []
    last_payload = {}
    for _, can_id, data in bucket:
        prev = last_payload.get(can_id)
        if prev is not None and len(prev) == len(data):
            deltas.append(sum(abs(a - b) for a, b in zip(prev, data)) / max(len(data), 1))
        last_payload[can_id] = data

    return {
        "frames_per_s": n / window_s,
        "unique_ids": float(len(unique_ids)),
        "unknown_id_frames": float(len(unknown)),
        "known_ids_seen": float(len(unique_ids & known)),
        "mean_gap_ms": mean_gap * 1000.0,
        "std_gap_ms": math.sqrt(var_gap) * 1000.0,
        "mean_dlc": (sum(len(d) for _, _, d in bucket) / n) if n else 0.0,
        "byte_entropy": _entropy(all_bytes),
        "mean_byte_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "max_byte": float(max(all_bytes)) if all_bytes else 0.0,
    }


FEATURE_NAMES = [
    "frames_per_s", "unique_ids", "unknown_id_frames", "known_ids_seen",
    "mean_gap_ms", "std_gap_ms", "mean_dlc", "byte_entropy",
    "mean_byte_delta", "max_byte",
]


def to_matrix(rows):
    """Rows of dicts to a list of lists, in a fixed column order."""
    return [[row[name] for name in FEATURE_NAMES] for row in rows]
