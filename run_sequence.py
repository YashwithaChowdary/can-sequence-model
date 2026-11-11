"""Train all three detectors on healthy traffic, then measure each on each fault.

This extends run.py with a third detector: a recurrent model that predicts the
next payload of each CAN id. The point of the experiment is not to find a
winner. It is to test one claim:

    the two faults the window-feature detectors nearly miss -- replay and
    drift -- are missed because those features describe the whole bus, and
    both faults change only the numbers inside a single message.

If the claim holds, the sequence model should catch replay and drift, and
should fail on the three faults the other two already handle, because a
flood and a fuzzer use ids it has no embedding for, and a silent node
produces no frames to predict at all.

    py run_sequence.py
    py run_sequence.py --epochs 20 --seconds 180 --target-fpr 0.05
"""
import argparse
import json

from can_bus import FAULTS, KNOWN_IDS, labelled_run, normal_traffic
from detect import ForestDetector, RuleBaseline, evaluate
from features import WINDOW_S, window_rows
from sequence_model import SequenceDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--epochs", type=int, default=30,
                    help="30 is where the loss stops falling quickly; it is "
                         "still decreasing slowly at that point")
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
    print("\nTraining the sequence model on healthy traffic only...")
    seq = SequenceDetector(
        KNOWN_IDS, target_fpr=args.target_fpr, epochs=args.epochs, seed=args.seed,
    ).fit(train_frames, calib_frames, window_s=args.window)

    print(f"\nAll three thresholds set at {args.target_fpr:.0%} false positives "
          f"on held-out healthy traffic.\n")

    header = (f"{'fault':<12}{'rules':>8}{'forest':>8}{'sequence':>10}"
              f"{'all three':>11}{'false alarms':>15}")
    print(header)
    print("-" * len(header))

    results = {}
    for fault in FAULTS:
        frames, fault_window = labelled_run(
            fault, seconds=args.seconds,
            start_s=args.seconds * 0.4, duration_s=args.seconds * 0.2,
            seed=args.seed + 7,
        )
        rows, times = window_rows(frames, KNOWN_IDS, args.window)

        r_pred = rules.predict(rows)
        f_pred = forest.predict(rows)
        s_pred = seq.predict(frames, window_s=args.window)
        # the sequence scores are built from the same window grid, but a run
        # can end mid-window, so trim to the shorter of the two to be safe
        n = min(len(r_pred), len(f_pred), len(s_pred), len(times))
        r_pred, f_pred, s_pred, t = r_pred[:n], f_pred[:n], s_pred[:n], times[:n]
        union = ((r_pred + f_pred + s_pred) > 0).astype(int)

        ev = {
            "rules": evaluate(r_pred, t, args.window, fault_window),
            "forest": evaluate(f_pred, t, args.window, fault_window),
            "sequence": evaluate(s_pred, t, args.window, fault_window),
            "all_three": evaluate(union, t, args.window, fault_window),
        }
        results[fault] = ev
        print(f"{fault:<12}"
              f"{ev['rules']['detection_rate']:>7.0%}"
              f"{ev['forest']['detection_rate']:>8.0%}"
              f"{ev['sequence']['detection_rate']:>10.0%}"
              f"{ev['all_three']['detection_rate']:>11.0%}"
              f"{ev['all_three']['false_alarm_rate']:>14.1%}")

    # Each detector is set to the same target individually, but the union of
    # three independent detectors raises the false alarm rate roughly to
    # 1 - (1 - fpr)^3. That is the price of the complementarity above and it
    # belongs in the table rather than in a footnote.
    print(f"\nFalse alarms per detector, outside the faulty window "
          f"(target {args.target_fpr:.0%} each):")
    print(f"{'fault':<12}{'rules':>8}{'forest':>8}{'sequence':>10}{'all three':>11}")
    for fault in FAULTS:
        ev = results[fault]
        print(f"{fault:<12}"
              f"{ev['rules']['false_alarm_rate']:>7.1%}"
              f"{ev['forest']['false_alarm_rate']:>8.1%}"
              f"{ev['sequence']['false_alarm_rate']:>10.1%}"
              f"{ev['all_three']['false_alarm_rate']:>11.1%}")

    print("\nWhich message the sequence model blamed, inside the faulty window:")
    for fault in FAULTS:
        frames, fault_window = labelled_run(
            fault, seconds=args.seconds,
            start_s=args.seconds * 0.4, duration_s=args.seconds * 0.2,
            seed=args.seed + 7,
        )
        pred = seq.predict(frames, window_s=args.window)
        blamed = seq.worst_id_per_window(frames, window_s=args.window)
        start, end = fault_window
        hits = [blamed[i] for i in range(min(len(pred), len(blamed)))
                if pred[i] and start <= i * args.window < end and blamed[i] is not None]
        if hits:
            top = max(set(hits), key=hits.count)
            print(f"  {fault:<12} 0x{top:03X}  "
                  f"({hits.count(top)}/{len(hits)} flagged windows)")
        else:
            print(f"  {fault:<12} nothing flagged")

    with open("results_sequence.json", "w") as fh:
        json.dump({"args": vars(args), "results": results,
                   "train_loss_byte_values": [l * 255.0 for l in seq.train_loss],
                   "threshold_byte_values": seq.threshold}, fh, indent=2)
    print("\nSaved results_sequence.json")


if __name__ == "__main__":
    main()
