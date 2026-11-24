"""Train-only PCA and ridge baseline for action readout from frozen DINO changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research.evaluate_robotwin_matched_state_probes import select_ridge, predict


def pooled_changes(cache):
    chunks = []
    for x, y in zip(cache["x"].split(8), cache["y"].split(8)):
        chunks.append(y.mean((1, 2)) - x.mean((1, 2)))
    return torch.cat(chunks).numpy().astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    train, val = [
        torch.load(p, weights_only=True, mmap=True)
        for p in (contract["external_probe"]["cache"], contract["cache"])
    ]
    train_ep = np.array([r["episode"] for r in train["rows"]])
    assert set(train_ep).isdisjoint(r["episode"] for r in val["rows"])
    train_diff, val_diff = pooled_changes(train), pooled_changes(val)
    state, vs = train["state"].numpy(), val["states"][:, 0, :16].numpy()
    target, vt = train["delta"].numpy(), val["delta"][:, :16].numpy()
    unique = np.unique(train_ep)
    np.random.default_rng(42).shuffle(unique)
    mapping = {ep: i % 5 for i, ep in enumerate(unique)}
    folds = np.array([mapping[ep] for ep in train_ep])
    alphas = [1e-5, 1e-3, 0.1, 1.0, 10.0]

    # Within each inner fold, learn the PCA on its own training rows.
    def features(fit_rows, query):
        data = train_diff[fit_rows]
        mean = data.mean(0)
        _, _, vt = np.linalg.svd(data - mean, full_matrices=False)
        return (data - mean) @ vt[:32].T, (query - mean) @ vt[:32].T

    from research.evaluate_robotwin_matched_state_probes import fit_ridge

    cv = np.zeros(len(alphas))
    for fold in range(5):
        hold = folds == fold
        z, vz = features(~hold, train_diff[hold])
        x = np.column_stack([state[~hold], z])
        vx = np.column_stack([state[hold], vz])
        for i, alpha in enumerate(alphas):
            pred = predict(fit_ridge(x, target[~hold], alpha), vx)
            cv[i] += ((pred - target[hold]) ** 2).sum()
    alpha = alphas[int(cv.argmin())]
    z, vz = features(np.ones(len(state), dtype=bool), val_diff)
    fitted = fit_ridge(np.column_stack([state, z]), target, alpha)
    preds = {
        "state_plus_dino_delta32": predict(fitted, np.column_stack([vs, vz])),
        "state_only": predict(select_ridge(state, target, folds, alphas)[0], vs),
    }
    groups = {"all": list(range(16)), **contract["state_groups"]}
    out = {
        name: {g: float(((p - vt)[:, cols] ** 2).mean()) for g, cols in groups.items()}
        for name, p in preds.items()
    }
    out.update(alpha=alpha, train_pairs=len(state), validation_pairs=len(vs))
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
