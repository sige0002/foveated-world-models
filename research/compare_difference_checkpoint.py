"""Compare a full difference-IDM checkpoint to archived standard-IDM predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from latent_action_model.core.lam_model import LatentLAMModel
from research.full_difference import subtract_current_reference
from research.evaluate_lawam_checkpoint import evaluate_model
from research.evaluate_robotwin_loss_ablation_fp32_v2 import digest, native_route_parity
from research.stage1_external_probe import run as probe


def paired(rows, baseline):
    ids = [(r["episode"], r["base"], r["donor_index"]) for r in rows]
    assert ids == [(r["episode"], r["base"], r["donor_index"]) for r in baseline]
    eps, group = np.unique([r["episode"] for r in rows], return_inverse=True)
    draws = np.random.default_rng(42).integers(len(eps), size=(10000, len(eps)))
    weights = np.bincount(group)[draws].sum(axis=1)
    metrics = {}
    for k in (
        "real_state_mse",
        "real_visual_mse",
        "real_position_state_mse",
        "real_orientation_state_mse",
        "real_gripper_state_mse",
        "swapped_state_mse",
        "swapped_visual_mse",
        "current_only_state_mse",
    ):
        a, b = [np.array([r["metrics"][k] for r in data]) for data in (rows, baseline)]
        samples = np.bincount(group, weights=a - b)[draws].sum(axis=1) / weights
        metrics[k] = dict(
            difference=a.mean(),
            standard=b.mean(),
            delta=(a - b).mean(),
            relative_pct=100 * (a.mean() / b.mean() - 1),
            ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        )
    return dict(episodes=len(eps), pairs=len(rows), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    run = args.checkpoint.parents[1]
    archive = Path("docs/robotwin-stage1-30k-20261117")
    contract = json.loads(
        (
            run.parent
            / "robotwin-official-fresh-b64-30k/evaluation-inputs/contract.json"
        ).read_text()
    )
    assert digest(contract["cache"]) == contract["cache_sha256"]
    assert digest(contract["split"]) == contract["split_sha256"]
    torch.set_num_threads(3)
    saved = torch.load(
        args.checkpoint, mmap=True, map_location="cpu", weights_only=False
    )
    step = saved["global_step"]
    ckpt_hash = digest(args.checkpoint)
    params = dict(saved["hyper_parameters"])
    params.pop("optimizer", None)
    model = LatentLAMModel(**params)
    model.load_state_dict(
        {k.removeprefix("lam."): v for k, v in saved["state_dict"].items()}, strict=True
    )
    model.encoder.register_forward_hook(subtract_current_reference)
    model.requires_grad_(False).cuda().eval()
    del saved
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    cache = torch.load(contract["cache"], mmap=True, weights_only=True)
    cache["state_groups"] = contract["state_groups"]
    split = json.loads(Path(contract["split"]).read_text())["ids"]
    eval_episodes = {r["episode"] for r in cache["rows"]}
    assert eval_episodes <= set(split["val"]) and eval_episodes.isdisjoint(
        split["train"]
    )
    ref = json.loads((archive / "results" / f"step{step:06d}.json").read_text())
    assert ref["complete"] and ref["identity"]["step"] == step
    with torch.inference_mode():
        parity = native_route_parity(model, cache, official=False)
        data = evaluate_model(model, cache, 2, out, official=False)
    report = dict(
        step=step,
        sha256=ckpt_hash,
        cache_sha256=contract["cache_sha256"],
        same_step=step,
        parity=parity,
        comparisons={step: paired(data["rows"], ref["rows"])},
    )
    (out / "rows.json").write_text(json.dumps(data["rows"]))
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print("Paired comparison complete", flush=True)
    assert (
        digest(contract["external_probe"]["cache"])
        == contract["external_probe"]["sha256"]
    )
    codes = torch.load(out / "posterior-codes.pt", weights_only=True)
    report["probe"] = probe(model, cache, codes, contract["split"])
    (out / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
