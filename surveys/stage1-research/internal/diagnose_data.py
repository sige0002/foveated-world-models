"""Read-only metadata/state diagnostics; no image decoding or model evaluation.

Run from repository root:
  .venv/bin/python surveys/stage1-research/internal/diagnose_data.py PREPROCESSED_DIR OUTPUT_JSON
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preprocessed_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "temporal_data", root / "vendor/dataset/temporal_data.py"
    )
    temporal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temporal)
    result = {
        "scope": "metadata/state only; no image or trained model evaluation",
        "horizon_seconds": 1.2,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        manifest = args.preprocessed_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        durations, tails, deltas, filled = [], [], [], []
        for row in rows:
            with np.load(args.preprocessed_dir / row["array_file"]) as arrays:
                times, states = arrays["timestamps"], arrays["states"]
                valid = arrays["original_valid"]
                for i, j, tail in temporal.timestamp_pairs(times, horizon=1.2):
                    durations.append(float(times[j] - times[i]))
                    tails.append(tail)
                    deltas.append(states[j] - states[i])
                    filled.append(not (valid[i].all() and valid[j].all()))
        delta = np.asarray(deltas)
        outside = np.abs(delta) > 1
        result["splits"][split] = {
            "episodes": len(rows),
            "pairs": len(durations),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "tail_pairs": sum(tails),
            "dt_quantiles_0_25_50_75_100": np.quantile(durations, [0, .25, .5, .75, 1]).tolist(),
            "filled_endpoint_pairs": sum(filled),
            "pairs_with_abs_state_delta_gt_1": int(outside.any(axis=1).sum()),
            "per_dimension_abs_delta_gt_1": outside.sum(axis=0).tolist(),
            "max_abs_state_delta": float(np.abs(delta).max()),
        }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
