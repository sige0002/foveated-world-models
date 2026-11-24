from pathlib import Path
import json
import math
import time
import torch
from lightning.pytorch import Callback


class DifferenceProgress(Callback):
    def on_train_start(self, trainer, model):
        self.start = time.monotonic()
        self.step = trainer.global_step

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        step = trainer.global_step
        loss = float(outputs["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        if step <= 3 or step % 10 == 0 or step == trainer.max_steps:
            now = time.monotonic()
            row = dict(
                step=step,
                loss=loss,
                seconds_per_update=(now - self.start) / (step - self.step),
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            with (Path(trainer.default_root_dir) / "progress.jsonl").open(
                "a"
            ) as stream:
                print(json.dumps(row), file=stream)
            self.start, self.step = now, step
