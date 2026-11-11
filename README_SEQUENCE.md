# A sequence model for the two faults window features cannot see

This extends [can-anomaly-detection](https://github.com/YashwithaChowdary/can-anomaly-detection)
with a third detector: a small recurrent network, in PyTorch, that learns what
payload each CAN message normally sends next.

The first project ended with an admission. A rule baseline and an Isolation
Forest over window features caught a bus flood, a fuzzer and a silent node, and
were **nearly blind to replay and drift**: 0% and 16% on replay, 0% and 1% on
drift. The README said why, and said what would be needed:

> *A replayed frame has a valid id, a plausible payload and roughly the right
> timing. A drifting signal changes nothing at all about the traffic pattern,
> only the number inside the payload. Window-level statistics cannot see
> either. Catching these needs per-signal models of what each byte normally
> does, which is a different and harder piece of work.*

This is that piece of work, and the claim turned out to be right.

## Results

120 s of traffic per run, 0.2 s windows, every detector trained on healthy
traffic only, every threshold set at a 2% false positive rate on a **separate**
healthy capture. Detection rate is the share of windows inside the fault that
were flagged.

| Fault | Rules | Isolation Forest | **Sequence** | All three |
|---|---|---|---|---|
| dos (bus flood) | 100% | 100% | 2% | 100% |
| fuzzing (random ids) | 100% | 62% | 63% | 100% |
| suspension (node goes silent) | 1% | 72% | 2% | 72% |
| **replay (stale frames re-sent)** | 0% | 16% | **88%** | 88% |
| **drift (signal slides out of range)** | 0% | 1% | **87%** | 88% |

False alarms outside the faulty window, each detector set to the same 2% target:

| Fault | Rules | Forest | Sequence | All three |
|---|---|---|---|---|
| dos | 0.0% | 2.7% | 1.9% | 4.6% |
| fuzzing | 0.0% | 2.7% | 3.5% | 6.2% |
| suspension | 0.0% | 2.7% | 1.9% | 4.6% |
| replay | 0.0% | 2.7% | 1.9% | 4.6% |
| drift | 0.0% | 2.7% | 2.9% | 5.6% |

Reproduce with `py run_sequence.py`. Numbers move by a few points with the seed.

### What this says

- **The gap was in the features, not in the model class.** Replay went from
  16% to 88% and drift from 1% to 87% without touching the faults, the
  windows, the training data or the false positive target. The only thing that
  changed is that the detector now looks at one message at a time instead of
  at aggregate statistics over the whole bus. A better classifier on the same
  ten window features would not have closed this gap.

- **The new detector is worse at everything the old ones were good at**, and
  that is not a defect. It scores 2% on the flood and 2% on the silent node,
  because a flood arrives on an id it has no embedding for and a silent node
  produces no frames to predict. It cannot say anything about traffic it has
  never seen, and it does not pretend to. The rule baseline owns those cases
  and costs nothing.

- **So the deployment is all three, and the union row is the real result**:
  100%, 100%, 72%, 88%, 88%. Each detector covers a different failure mode of
  the others.

- **Combining detectors costs false alarms, and the table says so.** Each one
  is set to 2% individually, but three independent detectors flag the union of
  their mistakes, which lands near 1 − 0.98³ ≈ 5.9%. Measured: 4.6% to 6.2%.
  That is the price of the coverage above, and on a real vehicle it is the
  number an engineer would argue about, so it belongs in the table and not in
  a footnote.

- **Fuzzing at 63% is a side effect rather than a success.** The fuzzer picks
  random ids across the whole 11-bit range, so it occasionally lands on one of
  the twelve known ids with a random payload, which the model finds absurd.
  The rules already catch fuzzing at 100% by noticing the unknown ids, so this
  column adds nothing useful.

## The model

A GRU with a per-id embedding, predicting the next payload from the previous
eight payloads **of that same id**.

```
previous 8 payloads, 8 scaled bytes each  ─┐
                                           ├─> GRU(16 -> 48) -> Linear(48 -> 8) -> sigmoid
id embedding, 8 dims, repeated per step   ─┘
```

9,992 parameters. Trained with L1 loss on Adam, 30 epochs, under a minute on a
laptop CPU — small enough that the honest comparison is against the rules and
the forest rather than against anything larger. The final training error is **1.76 byte values**, meaning the
eight bytes of the next payload are predicted to within about two values on
average. Loss was still falling slowly at 30 epochs.

Design choices worth defending:

- **One model for all twelve ids, with an embedding**, rather than twelve
  models. Fewer parameters, and ids that behave alike share what they learn.
  The cost is that an unseen id cannot be scored at all.
- **Next-payload prediction rather than reconstruction.** Both work as anomaly
  scores, but prediction error is the more natural question for a time series
  and easier to explain: the score is how far the actual bytes were from what
  should have come next.
- **L1 loss**, so the score is a mean absolute error in byte values and can be
  read directly. A score of 6 means "the bytes were on average six values away
  from the prediction".

## Why each message is scored against itself

This is the part that decides whether the detector works at all, and the first
version got it wrong.

Some messages are far easier to predict than others. Measured on healthy
training traffic, after 30 epochs:

| id | kind | median error | p90 error | scale used | max |
|---|---|---|---|---|---|
| 0x0A0 | counter | 0.49 | 11.10 | 10.61 | 31.57 |
| 0x0A1 | ramp | 0.44 | 1.05 | 0.61 | 1.63 |
| 0x0B0 | counter | 0.54 | 10.51 | 9.97 | 31.58 |
| 0x0B1 | ramp | 0.69 | 1.09 | 0.40 | 1.49 |
| 0x0C0 | ramp | 0.50 | 1.05 | 0.55 | 1.53 |
| 0x0C1 | constant | 0.36 | 0.38 | 0.05 | 0.40 |
| 0x0D0 | counter | 0.54 | 8.74 | 8.20 | 31.79 |
| 0x0D1 | constant | 0.53 | 0.55 | 0.05 | 0.55 |
| 0x0E0 | ramp | 0.53 | 1.19 | 0.66 | 2.07 |
| 0x0E1 | constant | 1.02 | 1.10 | 0.08 | 1.12 |
| 0x1F0 | counter | 0.73 | 8.67 | 7.94 | 31.81 |
| 0x1F1 | constant | 0.58 | 0.72 | 0.14 | 0.83 |

The counters wrap from 255 back to 0, and nothing in the history says exactly
when. They are predicted well most of the time — median error about 0.5 — but
every wrap produces an error above 30, which is perfectly normal behaviour for
that message.

My first attempt scored a window by the largest raw error in it. The threshold
came out at **31.6 byte values**, set entirely by counter wraps, and sat far
above anything a fault in a well behaved signal could produce. Detection on
drift was 5%. The model was fine; the scoring was wrong.

The fix is to judge each message against its own usual error. The score for a
frame is

```
(error − median error for this id) / (p90 error for this id − median)
```

so a score of 1.0 means "as far out as this message normally gets at its 90th
percentile". Median and p90 rather than mean and standard deviation, because
the wrap spikes would inflate both. A floor of 0.05 keeps a perfectly
predicted constant from dividing by nearly zero.

The window score is then the largest **per-id mean**, not the largest single
frame: averaging within an id and taking the worst id asks "is any one signal
behaving unusually", while averaging across the window would bury a fault in
one message out of twelve, and taking a single frame would fire on every wrap.

That mostly works. A wrap on 0x0A0 scores (31.57 − 0.49) / 10.61 ≈ 2.93, below
the 3.43 threshold on its own, and there are twenty frames of 0x0A0 in every
window to average it with, so it never gets near firing. The drift on 0x0E0
shifts one byte by up to 120 against a scale of 0.66, so it is never in doubt
in the other direction.

### Where it does not work, and why that is the interesting part

A wrap on **0x1F0**, the slowest counter, scores (31.81 − 0.73) / 7.94 ≈ 3.92,
which is *above* the threshold. Averaging cannot save it, because 0x1F0 sends
every 500 ms and a 0.2 s window therefore contains at most one of its frames.
There is nothing to average with.

That is not a hypothesis. Counting which message drove each false alarm over
240 s of healthy traffic the model had never seen, **all 29 of them came from
the two slowest counters**:

| message | period | frames per window | false alarms |
|---|---|---|---|
| 0x0D0 | 100 ms | 2.0 | 18 of 29 |
| 0x1F0 | 500 ms | 0.4 | 11 of 29 |
| 0x0A0 | 10 ms | 20.0 | 0 |
| 0x0B0 | 20 ms | 10.0 | 0 |

The fast counters wrap just as often and are predicted just as badly at the
wrap; they produce no false alarms at all because the window has enough of
their frames to dilute one bad one. The defence is not "averaging", it is
"averaging over enough samples", and below about two frames per window there
are not enough.

So the residual false alarm rate of this detector is almost entirely counter
wraps on slow messages, and the fix is to score over a fixed number of frames
per id rather than a fixed slice of time. That is the next thing I would
change, and it is a different change from the one the first project needed.

## It names the message

The Isolation Forest could only report which aggregate feature looked odd. This
model reports which arbitration id was worst predicted, which is the difference
between "something is wrong on the bus" and "0x0E0 is not doing what it
usually does".

| Fault | Message blamed | Agreement |
|---|---|---|
| replay | 0x0C0 | 106 of 106 flagged windows |
| drift | 0x0E0 | 103 of 104 flagged windows |
| fuzzing | 0x1F1 | 19 of 76 flagged windows |

0x0C0 and 0x0E0 are exactly the ids the simulator injects replay and drift
into. The fuzzing row is scattered, as it should be, since the fuzzer is not
attacking any particular message.

## Files

| File | What it does |
|---|---|
| `sequence_data.py` | Per-id payload sequences; per-id error statistics; window scoring |
| `sequence_model.py` | `PayloadPredictor` (GRU + id embedding) and `SequenceDetector` |
| `run_sequence.py` | Trains all three detectors and prints both tables above |
| `print_table.py` | Regenerates every measured number quoted in this README |
| `test_sequence.py` | 17 unit tests |

Plus the original `can_bus.py`, `features.py`, `detect.py`, `run.py` and
`test_detect.py` unchanged — the comparison would mean nothing if the baselines
had been touched.

## Run

    py -m pip install torch pandas scikit-learn numpy
    py run_sequence.py
    py run_sequence.py --epochs 50 --seconds 240 --target-fpr 0.05

## Test

    py -m unittest test_sequence.py      # 17 tests, about 11 s
    py -m unittest discover              # 41 tests with the originals

The tests on the trained model assert the direction of the result rather than
exact numbers, because those move with the seed. Two of them assert the claim
the project exists to make — that this detector beats the forest by at least 30
points on drift and on replay — so if that stops being true, the suite fails.
One asserts that it *misses* a flood, because a detector that quietly started
guessing about unknown ids would be worse, not better.

## Limits

- **The traffic is simulated, not recorded from a vehicle.** Real buses have
  bursts, bus-off events, diagnostic sessions and mode changes that this does
  not reproduce. A real counter may not be a clean `step % 256`, and a real
  ramp will be noisier, so both the error floor and the per-id scales would be
  larger. Every number here is a comparison between methods, not a performance
  claim for real traffic.
- **The drift and replay faults are synthetic and generous.** Drift moves one
  byte by up to 120 values, which is a long way. A slower or smaller drift
  would be harder, and finding where the detection rate falls off is the next
  experiment rather than a result I can claim.
- **Still no DBC decoding.** Payloads are raw bytes, so the model learns byte
  patterns rather than physical signals. A 16-bit signal split across two
  bytes is learned as two weakly related bytes, which is almost certainly why
  the error floor sits at 1.76 rather than lower.
- **No id can be scored on first sight**, and the first eight frames of every
  id are context rather than samples. On a bus where ids appear and disappear
  with mode changes, that matters more than it does here.
- **Thirty epochs was chosen by watching the loss stop falling quickly, not by
  a validation sweep.** The loss was still decreasing slowly when training
  stopped.
