"""Print the per-message error table and the numbers the README quotes.

Everything in README_SEQUENCE.md that is a measurement comes from here, so it
can be regenerated rather than trusted.

    py print_table.py
    py print_table.py --epochs 30 --seconds 120
"""
import argparse

import numpy as np

from can_bus import KNOWN_IDS, MESSAGES, normal_traffic
from sequence_model import SequenceDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = normal_traffic(seconds=args.seconds, seed=args.seed)
    calib = normal_traffic(seconds=args.seconds, seed=args.seed + 101)
    seq = SequenceDetector(
        KNOWN_IDS, target_fpr=args.target_fpr, epochs=args.epochs, seed=args.seed,
    ).fit(train, calib, verbose=False)

    errors, _, id_index = seq.frame_errors(train)
    kind = {m[0]: m[4] for m in MESSAGES}

    print("\n## Per-message error table, ready to paste into the README\n")
    print("| id | kind | median error | p90 error | scale used | max |")
    print("|---|---|---|---|---|---|")
    rows = []
    for i, can_id in enumerate(KNOWN_IDS):
        mine = errors[id_index == i]
        if len(mine) == 0:
            continue
        median = seq.centre[i]
        p90 = float(np.quantile(mine, 0.90))
        scale = seq.scale[i]
        worst = float(mine.max())
        rows.append((can_id, kind[can_id], median, p90, scale, worst))
        print(f"| 0x{can_id:03X} | {kind[can_id]} | {median:.2f} | "
              f"{p90:.2f} | {scale:.2f} | {worst:.2f} |")

    print("\n## Sentences in the README that quote a number\n")
    print(f"final training error        {seq.train_loss[-1] * 255.0:.2f} byte values")
    print(f"threshold                   {seq.threshold:.2f}")

    lowest = min(rows, key=lambda r: r[2])
    highest = max(rows, key=lambda r: r[2])
    print(f"normal error range          {lowest[2]:.2f} (0x{lowest[0]:03X}) "
          f"to {highest[2]:.2f} (0x{highest[0]:03X})")

    # What a single counter wrap scores, and whether the window has enough
    # frames of that id to average it away. This is the whole false alarm
    # story, so it is printed rather than asserted.
    period = {m[0]: m[1] for m in MESSAGES}
    counters = [r for r in rows if r[1] == "counter"]
    if counters:
        print(f"\n## What one counter wrap scores, against a threshold of "
              f"{seq.threshold:.2f}\n")
        print("| message | period | frames per window | worst wrap | scores | fires alone |")
        print("|---|---|---|---|---|---|")
        for can_id, _, median, _, scale, worst in sorted(
                counters, key=lambda r: period[r[0]]):
            score = (worst - median) / scale
            per_window = 200.0 / period[can_id]
            print(f"| 0x{can_id:03X} | {period[can_id]} ms | {per_window:.1f} | "
                  f"{worst:.2f} | {score:.2f} | "
                  f"{'yes' if score > seq.threshold else 'no'} |")
        print("\nA wrap that scores above the threshold only causes a false "
              "alarm if the\nwindow has too few frames of that id to average "
              "it away, which is why the\nslow counters are the ones that fire.")

    # Where the false alarms actually come from, on healthy traffic the model
    # has never seen.
    held = normal_traffic(seconds=args.seconds * 2, seed=args.seed + 555)
    pred = seq.predict(held)
    blamed = seq.worst_id_per_window(held)
    counts = {}
    for i in range(min(len(pred), len(blamed))):
        if pred[i] and blamed[i] is not None:
            counts[blamed[i]] = counts.get(blamed[i], 0) + 1
    total = sum(counts.values())
    print(f"\n## False alarms on {args.seconds * 2:.0f}s of unseen healthy "
          f"traffic: {int(pred.sum())} of {len(pred)} windows "
          f"({pred.mean():.1%})\n")
    if total:
        print("| message | kind | period | frames per window | false alarms |")
        print("|---|---|---|---|---|")
        for can_id, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"| 0x{can_id:03X} | {kind[can_id]} | {period[can_id]} ms | "
                  f"{200.0 / period[can_id]:.1f} | {n} of {total} |")
    else:
        print("none")


if __name__ == "__main__":
    main()
