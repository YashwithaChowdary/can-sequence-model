"""A small simulator for CAN bus traffic, plus five kinds of fault.

Real CAN traffic is mostly periodic: each electronic control unit sends a
message with a fixed arbitration ID at a fixed interval, with a little jitter.
That regularity is what makes anomaly detection possible, and it is also why
some faults are easy to see and others are nearly invisible.

A frame is (timestamp_s, can_id, list of data bytes).

No real vehicle data is used here. The point of the simulator is to produce
traffic with the right shape so that a detector can be built and measured
before real logs are available.
"""
import random

# ---------------------------------------------------------------------------
# The bus description: twelve periodic messages, as a small vehicle might have.
# period_ms, jitter_ms, payload length, and how each payload behaves.
# ---------------------------------------------------------------------------
MESSAGES = [
    # can_id, period_ms, jitter_ms, dlc, kind
    (0x0A0, 10, 0.4, 8, "counter"),    # wheel speed, fast
    (0x0A1, 10, 0.4, 8, "ramp"),       # engine rpm
    (0x0B0, 20, 0.8, 8, "counter"),    # steering angle
    (0x0B1, 20, 0.8, 6, "ramp"),       # throttle position
    (0x0C0, 50, 1.5, 8, "ramp"),       # brake pressure
    (0x0C1, 50, 1.5, 4, "constant"),   # gear state
    (0x0D0, 100, 3.0, 8, "counter"),   # battery state
    (0x0D1, 100, 3.0, 2, "constant"),  # door status
    (0x0E0, 200, 5.0, 8, "ramp"),      # coolant temperature
    (0x0E1, 200, 5.0, 3, "constant"),  # lamp status
    (0x1F0, 500, 10.0, 8, "counter"),  # diagnostics
    (0x1F1, 1000, 20.0, 8, "constant"),  # vin / static info
]

KNOWN_IDS = [m[0] for m in MESSAGES]

FAULTS = ["dos", "fuzzing", "suspension", "replay", "drift"]


def _payload(kind, step, dlc, rng):
    """Bytes for one frame. Counters wrap, ramps drift slowly, constants hold."""
    if kind == "counter":
        base = [(step + i * 7) % 256 for i in range(dlc)]
    elif kind == "ramp":
        value = int(128 + 100 * (0.5 - abs(((step / 400.0) % 1.0) - 0.5)) )
        base = [value] + [(value + i) % 256 for i in range(1, dlc)]
    else:  # constant
        base = [(0x40 + i) for i in range(dlc)]
    # a little noise on the last byte, as real signals have
    if dlc:
        base[-1] = (base[-1] + rng.randint(0, 2)) % 256
    return base


def normal_traffic(seconds=60.0, seed=0):
    """Frames for `seconds` of healthy bus traffic, sorted by time."""
    rng = random.Random(seed)
    frames = []
    for can_id, period_ms, jitter_ms, dlc, kind in MESSAGES:
        t = rng.uniform(0, period_ms / 1000.0)
        step = 0
        while t < seconds:
            frames.append((t, can_id, _payload(kind, step, dlc, rng)))
            t += (period_ms + rng.gauss(0, jitter_ms)) / 1000.0
            step += 1
    frames.sort(key=lambda f: f[0])
    return frames


def inject(frames, fault, start_s, duration_s, seed=0):
    """Return a copy of `frames` with one fault applied over a time window.

    dos         a flood of a high-priority id, the classic bus denial of service
    fuzzing     random ids with random payloads, as a fuzzer or a broken node
    suspension  one ecu stops transmitting
    replay      frames of one id are re-sent out of phase, payloads stale
    drift       one signal's value slides out of its normal range, timing normal
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}, expected one of {FAULTS}")
    rng = random.Random(seed)
    end_s = start_s + duration_s
    out = [f for f in frames]

    if fault == "dos":
        t = start_s
        while t < end_s:
            out.append((t, 0x000, [0x00] * 8))
            t += 0.0005  # 2000 frames per second
    elif fault == "fuzzing":
        t = start_s
        while t < end_s:
            can_id = rng.randrange(0x000, 0x7FF)
            dlc = rng.randint(0, 8)
            out.append((t, can_id, [rng.randrange(256) for _ in range(dlc)]))
            t += rng.expovariate(300.0)
    elif fault == "suspension":
        victim = 0x0B0
        out = [f for f in out if not (f[1] == victim and start_s <= f[0] < end_s)]
    elif fault == "replay":
        victim = 0x0C0
        source = [f for f in out if f[1] == victim and f[0] < start_s][-20:]
        if source:
            t = start_s
            i = 0
            while t < end_s:
                out.append((t, victim, list(source[i % len(source)][2])))
                t += 0.05
                i += 1
    elif fault == "drift":
        victim = 0x0E0
        shifted = []
        for f in out:
            if f[1] == victim and start_s <= f[0] < end_s:
                frac = (f[0] - start_s) / max(duration_s, 1e-9)
                payload = list(f[2])
                payload[0] = min(255, int(payload[0] + 120 * frac))
                shifted.append((f[0], f[1], payload))
            else:
                shifted.append(f)
        out = shifted

    out.sort(key=lambda f: f[0])
    return out


def labelled_run(fault, seconds=60.0, start_s=20.0, duration_s=20.0, seed=0):
    """Traffic with one fault, plus the (start, end) of the faulty window."""
    frames = normal_traffic(seconds=seconds, seed=seed)
    return inject(frames, fault, start_s, duration_s, seed=seed), (start_s, start_s + duration_s)
