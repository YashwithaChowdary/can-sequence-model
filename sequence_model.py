"""A small recurrent model that learns what each CAN message does next.

The idea is narrow on purpose. For one arbitration id, the previous eight
payloads are a strong clue to the ninth: a counter increments, a ramp carries
on in the direction it was going, a constant stays put. A model that has
learned those habits from healthy traffic will predict the next payload
closely, and the size of its mistake is an anomaly score.

Why that should see what window features cannot:

  replay   the payload is re-sent from earlier in the log, so it is a valid
           value for the signal but the wrong one for this moment. The gap
           between predicted and actual is large even though nothing about
           the traffic shape has changed.
  drift    one byte slides out of its usual range. The model has never seen
           that value follow this history, so the error grows as the drift
           grows.

One model is shared across all twelve ids rather than one model per id, with
an embedding standing in for the id. That keeps the parameter count small, and
it lets ids that behave alike share what they have learned. The cost is that
an id with no embedding -- one the model has never seen -- cannot be scored at
all, which is a genuine limitation and is reported as such.

The score is a mean absolute error, converted back into byte values, so a
score of 6.0 means "the eight bytes were on average six values away from
what was expected". That is a quantity an engineer can argue with.
"""
import numpy as np
import torch
from torch import nn

from sequence_data import (
    BYTE_SCALE, HISTORY, MAX_DLC, build_sequences, n_windows_for,
    per_id_error_stats, window_scores,
)


class PayloadPredictor(nn.Module):
    """Predict the next payload of a CAN id from its previous payloads."""

    def __init__(self, n_ids, embed_dim=8, hidden=48, n_bytes=MAX_DLC):
        super().__init__()
        self.embed = nn.Embedding(n_ids, embed_dim)
        self.gru = nn.GRU(n_bytes + embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_bytes)

    def forward(self, history, id_index):
        # history (batch, HISTORY, MAX_DLC), id_index (batch,)
        e = self.embed(id_index)                                  # (batch, embed)
        e = e.unsqueeze(1).expand(-1, history.size(1), -1)        # per time step
        x = torch.cat([history, e], dim=-1)
        _, last_hidden = self.gru(x)
        # sigmoid because the targets are bytes scaled to 0..1
        return torch.sigmoid(self.head(last_hidden[-1]))


class SequenceDetector:
    """Trains on healthy traffic only; threshold from separate healthy traffic.

    Deliberately the same contract as ForestDetector in detect.py, so the
    three detectors can be compared on identical runs with identical windows.
    """

    def __init__(self, known_ids, target_fpr=0.02, epochs=30, hidden=48,
                 batch_size=256, lr=1e-3, seed=0):
        self.known_ids = list(known_ids)
        self.target_fpr = target_fpr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.threshold = None
        self.centre = None      # per-id normal error level, set during fit
        self.scale = None       # per-id spread of normal error, set during fit
        self.train_loss = []
        torch.manual_seed(seed)
        self.model = PayloadPredictor(len(self.known_ids), hidden=hidden)

    # -- training ----------------------------------------------------------
    def fit(self, train_frames, calib_frames, window_s=0.2, verbose=True):
        data = build_sequences(train_frames, self.known_ids)
        if len(data["time"]) == 0:
            raise ValueError("no scorable frames in the training traffic")

        history = torch.from_numpy(data["history"])
        ids = torch.from_numpy(data["id_index"])
        target = torch.from_numpy(data["target"])

        loss_fn = nn.L1Loss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(ids)
        generator = torch.Generator().manual_seed(self.seed)

        self.model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            running, batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                pred = self.model(history[idx], ids[idx])
                loss = loss_fn(pred, target[idx])
                loss.backward()
                opt.step()
                running += loss.item()
                batches += 1
            mean_loss = running / max(batches, 1)
            self.train_loss.append(mean_loss)
            if verbose:
                print(f"  epoch {epoch + 1:2d}/{self.epochs}  "
                      f"mean abs error {mean_loss * BYTE_SCALE:5.2f} byte values")

        # How well is each id normally predicted? Measured on the training
        # traffic, so that each message is later judged against itself rather
        # than against the hardest message on the bus.
        errors, _, id_index = self.frame_errors(train_frames)
        self.centre, self.scale = per_id_error_stats(
            errors, id_index, len(self.known_ids),
        )
        if verbose:
            worst = int(np.argmax(self.centre))
            best = int(np.argmin(self.centre))
            print(f"  normal error ranges from {self.centre[best]:.2f} "
                  f"(0x{self.known_ids[best]:03X}) to {self.centre[worst]:.2f} "
                  f"byte values (0x{self.known_ids[worst]:03X}); "
                  f"each id is scored against its own")

        # Threshold on a *separate* healthy capture, never on faulty data.
        calib = self.window_scores(calib_frames, window_s=window_s)
        self.threshold = float(np.quantile(calib, 1.0 - self.target_fpr))
        if verbose:
            print(f"  threshold {self.threshold:.2f} "
                  f"({self.target_fpr:.0%} of healthy windows above it)")
        return self

    # -- scoring -----------------------------------------------------------
    def frame_errors(self, frames):
        """Per-frame prediction error in byte values, with ids and times."""
        data = build_sequences(frames, self.known_ids)
        if len(data["time"]) == 0:
            empty = np.zeros(0)
            return empty, empty, np.zeros(0, dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                torch.from_numpy(data["history"]),
                torch.from_numpy(data["id_index"]),
            ).numpy()
        errors = np.abs(pred - data["target"]).mean(axis=1) * BYTE_SCALE
        return errors, data["time"], data["id_index"]

    def window_scores(self, frames, window_s=0.2):
        """One score per window: the worst id's mean standardised error.

        A score of 1.0 means some message was as far from its prediction as
        it normally gets at its 90th percentile. A score of 10 means a message
        is far outside anything seen while healthy.
        """
        errors, times, id_index = self.frame_errors(frames)
        return window_scores(
            errors, times, id_index, n_windows_for(frames, window_s), window_s,
            centre=self.centre, scale=self.scale,
        )

    def predict(self, frames, window_s=0.2):
        if self.threshold is None:
            raise RuntimeError("fit() must be called before predict()")
        return (self.window_scores(frames, window_s) > self.threshold).astype(int)

    def worst_id_per_window(self, frames, window_s=0.2):
        """Which arbitration id was worst predicted in each window, or None.

        The forest can only say which aggregate feature looked odd. This can
        name the message, which is the difference between "something is wrong
        on the bus" and "0x0E0 is not doing what it usually does".
        """
        errors, times, id_index = self.frame_errors(frames)
        n = n_windows_for(frames, window_s)
        best = [(-1e9, None)] * n
        for err, t, idx in zip(errors, times, id_index):
            i = int(idx)
            # the same standardisation the score uses, so the id that is
            # blamed is the id that actually drove the score
            if self.centre is not None and self.scale is not None:
                err = (err - self.centre[i]) / self.scale[i]
            w = int(t / window_s)
            if 0 <= w < n and err > best[w][0]:
                best[w] = (err, self.known_ids[i])
        return [can_id for _, can_id in best]
