"""Train on healthy traffic, then measure both detectors on each fault.

    py run.py
    py run.py --target-fpr 0.05 --seconds 120
"""
import argparse
import json

from can_bus import FAULTS, KNOWN_IDS, labelled_run, normal_traffic
from detect import ForestDetector, RuleBaseline, evaluate
from features import WINDOW_S, window_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="length of each traffic capture in seconds")
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="false positive rate the threshold is set to")
    ap.add_argument("--window", type=float, default=WINDOW_S,
                    help="window length in seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Collecting healthy traffic for training and calibration...")
    train_frames = normal_traffic(seconds=args.seconds, seed=args.seed)
    calib_frames = normal_traffic(seconds=args.seconds, seed=args.seed + 101)
    train_rows, _ = window_rows(train_frames, KNOWN_IDS, args.window)
    calib_rows, _ = window_rows(calib_frames, KNOWN_IDS, args.window)
    print(f"  {len(train_frames)} frames -> {len(train_rows)} training windows")
    print(f"  {len(calib_frames)} frames -> {len(calib_rows)} calibration windows")

    rules = RuleBaseline().fit(train_frames, train_rows)
    forest = ForestDetector(target_fpr=args.target_fpr).fit(train_rows, calib_rows)
    print(f"  rule baseline: {len(rules.known_ids)} known ids, "
          f"max healthy rate {rules.max_rate:.0f} frames/s")
    print(f"  isolation forest threshold set at {args.target_fpr:.0%} "
          f"false positives on held-out healthy traffic\n")

    header = (f"{'fault':<12}{'rules':>10}{'forest':>10}{'both':>10}"
              f"{'false alarms (both)':>22}")
    print(header)
    print("-" * len(header))

    results = {}
    for fault in FAULTS:
        frames, window = labelled_run(
            fault, seconds=args.seconds,
            start_s=args.seconds * 0.4, duration_s=args.seconds * 0.2,
            seed=args.seed + 7,
        )
        rows, times = window_rows(frames, KNOWN_IDS, args.window)
        r_pred = rules.predict(rows)
        f_pred = forest.predict(rows)
        both = ((r_pred + f_pred) > 0).astype(int)
        r_eval = evaluate(r_pred, times, args.window, window)
        f_eval = evaluate(f_pred, times, args.window, window)
        b_eval = evaluate(both, times, args.window, window)
        results[fault] = {"rules": r_eval, "forest": f_eval, "both": b_eval}
        print(f"{fault:<12}"
              f"{r_eval['detection_rate']:>9.0%}"
              f"{f_eval['detection_rate']:>10.0%}"
              f"{b_eval['detection_rate']:>10.0%}"
              f"{b_eval['false_alarm_rate']:>21.1%}")

    print("\nWhat the forest pointed at, for the windows it flagged:")
    for fault in FAULTS:
        frames, window = labelled_run(
            fault, seconds=args.seconds,
            start_s=args.seconds * 0.4, duration_s=args.seconds * 0.2,
            seed=args.seed + 7,
        )
        rows, times = window_rows(frames, KNOWN_IDS, args.window)
        pred = forest.predict(rows)
        hints = forest.feature_importance_hint(rows)
        start, end = window
        flagged = [hints[i] for i, t in enumerate(times)
                   if pred[i] and start <= t < end]
        if flagged:
            top = max(set(flagged), key=flagged.count)
            print(f"  {fault:<12} {top} "
                  f"({flagged.count(top)}/{len(flagged)} flagged windows)")
        else:
            print(f"  {fault:<12} nothing flagged")

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
