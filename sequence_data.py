"""Turn CAN frames into per-ID payload sequences.

The window features in `features.py` describe a slice of the whole bus: how
many frames arrived, how many distinct ids, how regular the gaps were. Those
are the right features for a flood, for an unknown id, or for a node that goes
quiet, because all three change the shape of the traffic.

They are the wrong features for a fault that changes only the numbers inside
one message. A replayed frame arrives on time, with a valid id and a payload
that is a perfectly ordinary value for that signal -- just not the value that
should come next. A drifting signal changes nothing about the traffic at all.
Aggregated over a window and mixed with eleven other messages, neither leaves
a mark.

So this module looks at one id at a time. For every frame of a known id it
collects the previous `HISTORY` payloads of that same id, which is the context
a model needs in order to say what should come next.

Bytes are scaled to 0..1 by dividing by 255, so a prediction error of 0.01 is
about two and a half byte values. Payloads shorter than eight bytes are padded
with zeros; the model learns each id's real length through its embedding.
"""
from collections import defaultdict, deque

import numpy as np

MAX_DLC = 8
HISTORY = 8          # how many previous payloads of the same id the model sees
BYTE_SCALE = 255.0


def _pad(data):
    """A payload as a fixed-length vector of eight scaled bytes."""
    out = np.zeros(MAX_DLC, dtype=np.float32)
    for i, b in enumerate(data[:MAX_DLC]):
        out[i] = b / BYTE_SCALE
    return out


def build_sequences(frames, known_ids):
    """Per-id sequences: for each frame, the previous HISTORY payloads of that id.

    Frames whose id is not in `known_ids` are skipped, and so are the first
    HISTORY frames of each id, because there is not yet any history to learn
    from. That is a real limitation rather than an oversight: this model can
    say nothing at all about an id it has never seen, which is precisely the
    case the rule baseline already handles perfectly.

    Returns a dict of numpy arrays:
        id_index  (n,)                 position of the id in `known_ids`
        history   (n, HISTORY, MAX_DLC) the previous payloads, oldest first
        target    (n, MAX_DLC)         the payload that actually arrived
        time      (n,)                 timestamp of the target frame
    """
    index_of = {can_id: i for i, can_id in enumerate(known_ids)}
    recent = defaultdict(lambda: deque(maxlen=HISTORY))

    id_index, history, target, time = [], [], [], []
    for t, can_id, data in frames:
        if can_id not in index_of:
            continue
        past = recent[can_id]
        if len(past) == HISTORY:
            id_index.append(index_of[can_id])
            history.append(np.stack(past))
            target.append(_pad(data))
            time.append(t)
        past.append(_pad(data))

    if not id_index:
        return {
            "id_index": np.zeros(0, dtype=np.int64),
            "history": np.zeros((0, HISTORY, MAX_DLC), dtype=np.float32),
            "target": np.zeros((0, MAX_DLC), dtype=np.float32),
            "time": np.zeros(0, dtype=np.float64),
        }

    return {
        "id_index": np.array(id_index, dtype=np.int64),
        "history": np.stack(history).astype(np.float32),
        "target": np.stack(target).astype(np.float32),
        "time": np.array(time, dtype=np.float64),
    }


def per_id_error_stats(errors, id_index, n_ids, floor=0.05):
    """How well each id is normally predicted, measured on healthy traffic.

    This matters more than it sounds. Some messages are far easier to predict
    than others: a constant never moves, a ramp carries on, but a counter
    wraps from 255 back to 0 and nothing in the history says exactly when.
    Measured on healthy traffic, the ramps and constants here sit around half
    a byte value of error while the counters average three and spike past
    thirty at every wrap.

    So a single global threshold on raw error is useless: it gets set by the
    counter wraps, which are perfectly normal, and ends up far above anything
    a fault in a well behaved signal could produce. Each id has to be judged
    against its own usual error instead.

    Returns (centre, scale) per id, using the median and the gap up to the
    90th percentile -- robust to the wrap spikes, unlike a mean and a standard
    deviation. `floor` keeps a perfectly predicted constant from dividing by
    something near zero.
    """
    centre = np.zeros(n_ids, dtype=np.float64)
    scale = np.full(n_ids, floor, dtype=np.float64)
    for i in range(n_ids):
        mine = errors[id_index == i]
        if len(mine) < 2:
            continue
        med = float(np.median(mine))
        centre[i] = med
        scale[i] = max(float(np.quantile(mine, 0.90)) - med, floor)
    return centre, scale


def window_scores(errors, times, id_index, n_windows, window_s,
                  centre=None, scale=None):
    """Collapse per-frame prediction errors into one score per time window.

    Two deliberate choices here.

    First, each frame's error is standardised against its own id's normal
    error level, for the reason described in `per_id_error_stats`. The score
    is then "how unusual is this message for itself", not "how large is the
    error in bytes".

    Second, the window score is the largest *per-id mean*, not the largest
    single frame. Drift affects one message out of twelve, so averaging over
    the whole window would bury it -- but averaging within an id and then
    taking the worst id asks the right question: is any one signal behaving
    unusually right now? It also stops a single counter wrap from setting off
    an alarm, because it is averaged with that id's other frames in the same
    window.

    Windows with no scorable frame score 0.0, the most normal value, so they
    are never flagged. Those are windows made up entirely of unknown ids, or
    where a message has gone silent -- the cases the other two detectors own.
    """
    out = np.zeros(n_windows, dtype=np.float64)
    if len(errors) == 0:
        return out

    if centre is not None and scale is not None:
        standardised = (errors - centre[id_index]) / scale[id_index]
    else:
        standardised = errors

    sums = defaultdict(float)
    counts = defaultdict(int)
    for s, t, i in zip(standardised, times, id_index):
        w = int(t / window_s)
        if 0 <= w < n_windows:
            key = (w, int(i))
            sums[key] += s
            counts[key] += 1

    for key, total in sums.items():
        window, _ = key
        mean = total / counts[key]
        if mean > out[window]:
            out[window] = mean
    return out


def n_windows_for(frames, window_s):
    """The same window count that features.window_rows produces, so that the
    sequence scores line up with the window feature rows one for one."""
    if not frames:
        return 0
    return int(frames[-1][0] / window_s) + 1
