"""Stop only after consecutive strict increases in validation loss."""
from __future__ import annotations

import math

from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState


class ConsecutiveWorseningCallback(TrainerCallback, ExportableState):
    """Compare with the previous loss; a tie or improvement resets the streak."""

    def __init__(self, patience=2, metric_name="eval_loss"):
        if type(patience) is not int or patience < 1:
            raise ValueError("Patience must be a positive integer")
        self.patience = patience
        self.metric_name = metric_name
        self.previous_loss = None
        self.consecutive_increases = 0

    def on_train_begin(self, args, state, control, **kwargs):
        metric = args.metric_for_best_model
        if isinstance(metric, str) and not metric.startswith("eval_"):
            metric = "eval_" + metric
        if not args.load_best_model_at_end or args.greater_is_better is not False or metric != self.metric_name:
            raise ValueError("Loss stopping requires best-model loading with eval_loss and lower-is-better")
        if self.previous_loss is not None and not math.isfinite(self.previous_loss):
            raise ValueError("Restored validation loss is not finite")
        if type(self.consecutive_increases) is not int or self.consecutive_increases < 0:
            raise ValueError("Invalid restored loss-increase counter")
        if self.consecutive_increases >= self.patience:
            control.should_training_stop = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            raise ValueError(f"Missing validation metric {self.metric_name!r}")
        value = metrics[self.metric_name]
        if isinstance(value, bool):
            raise ValueError("Validation loss cannot be boolean")
        loss = float(value)
        if not math.isfinite(loss):
            raise ValueError("Validation loss is not finite")
        if self.previous_loss is not None and loss > self.previous_loss:
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0
        self.previous_loss = loss
        if self.consecutive_increases >= self.patience:
            control.should_training_stop = True
        return control

    def state(self):
        return {"args": {"patience": self.patience, "metric_name": self.metric_name},
                "attributes": {"previous_loss": self.previous_loss,
                               "consecutive_increases": self.consecutive_increases}}
