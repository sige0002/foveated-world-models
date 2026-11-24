"""Nested episode-wise CV on existing validation posterior codes for both frozen models."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research.evaluate_robotwin_matched_state_probes import select_ridge, predict


def folds(episodes, seed):
    unique, inverse = np.unique(episodes, return_inverse=True)
    return (np.random.default_rng(seed).permutation(len(unique)) % 5)[inverse]


def crossfit(x, target, episodes):
    outer = folds(episodes, 8193)
    pred = np.empty_like(target)
    for i in range(5):
        subset = outer != i
        fit, _ = select_ridge(
            x[subset],
            target[subset],
            folds(episodes[subset], 510 + i),
            [1e-5, 1e-3, 0.1, 1.0, 10.0],
        )
        pred[~subset] = predict(fit, x[~subset])
    return (pred - target) ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument(
        "--report-dir", type=Path, help="Directory with the paired comparison report"
    )
    args = parser.parse_args()
    out = args.eval_dir
    report_dir = args.report_dir or out
    report = json.loads((report_dir / "comparison.json").read_text())
    baseline = out.parents[1] / "robotwin-official-fresh-b64-30k"
    contract = json.loads((baseline / "evaluation-inputs/contract.json").read_text())
    cache = torch.load(contract["cache"], weights_only=True, mmap=True)
    episodes = np.array([r["episode"] for r in cache["rows"]])
    state = cache["states"][:, 0, :16].numpy().astype(float)
    target = cache["delta"][:, :16].numpy().astype(float)
    groups = {"all": list(range(16)), **contract["state_groups"]}
    models = {"state_only": crossfit(state, target, episodes)}
    sources = {"difference": out / "posterior-codes.pt"}
    for step in report["comparisons"]:
        candidates = list(
            (baseline / "evaluations").glob(f"step{int(step):06d}-*/posterior-codes.pt")
        )
        assert len(candidates) == 1
        sources["standard_" + step] = candidates[0]
    for name, path in sources.items():
        codes = torch.load(path, weights_only=True)
        assert codes["row_identity_sha256"] == cache["reference_identity_sha256"]
        z = codes["mu"].flatten(1).numpy()
        models[name] = crossfit(np.concatenate([state, z], axis=1), target, episodes)
    eps, group = np.unique(episodes, return_inverse=True)
    indices = np.random.default_rng(50).integers(len(eps), size=(10000, len(eps)))
    den = np.bincount(group)[indices].sum(1)
    summary = {}
    for name, errors in models.items():
        row = {}
        for g, cols in groups.items():
            loss = errors[:, cols].mean(1)
            delta = models["difference"][:, cols].mean(1) - loss
            pooled = np.bincount(group, weights=delta)[indices].sum(1) / den
            row[g] = dict(
                mse=loss.mean(),
                diff_minus_this=delta.mean(),
                paired_ci95=np.quantile(pooled, [0.025, 0.975]).tolist(),
            )
        summary[name] = row
    (report_dir / "crossfit.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
