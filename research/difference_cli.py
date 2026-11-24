"""Run the full RoboTwin difference-IDM experiment on the SSD1 workstation."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STORE = Path("/media/sadasue/SSD1/foveated-world-models")
AUDIT = STORE / "research/official-data-contract-audit-20261113"
RUNS = STORE / "runs"
SPLIT = RUNS / "robotwin-loss-ablation-b4-20261114/kl_mse/episode_split.json"
SOURCE = "c12168d9077edea67fe7c1b70d08b80b8e65a9c8"
CURRENT = "stage1-difference-30k-20261124"


def run_path(name):
    if not name or name.startswith(".") or Path(name).name != name:
        raise ValueError("Specify one run directory name, without a path")
    path = RUNS / name
    if path.is_symlink():
        raise ValueError(f"Expected a run directory, found symlink: {path}")
    return path


def status(name):
    folder = run_path(name)
    progress = folder / "progress.jsonl"
    if not progress.exists():
        raise ValueError(f"No progress recorded: {progress}")
    last = progress.read_text().splitlines()[-1]
    print(folder.resolve(), last, sep="\n")
    for path in sorted(folder.glob("metrics/version_*/metrics.csv")):
        with path.open() as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("val_loss")]
        if rows:
            print(
                path.parent.name,
                {k: rows[-1][k] for k in ("step", "val_loss", "val/state_loss")},
            )
    print(
        "Checkpoints:",
        ", ".join(p.name for p in sorted(folder.glob("checkpoints/*.ckpt"))),
    )
    if name == CURRENT:
        subprocess.run(
            ["systemctl", "--user", "is-active", "lawam-difference-30k-20261124"],
            check=False,
        )


def check():
    for p in (
        RUNS,
        SPLIT,
        AUDIT / "LaWAM/starVLA",
        AUDIT / "python-deps",
        AUDIT / "pytorch3d",
        AUDIT / "robotwin-qfix-data/robotwin_merged/videos",
        STORE / "models/dinov3-vitb16-pretrain-lvd1689m/config.json",
    ):
        if not p.exists():
            raise ValueError(f"Missing training input: {p}")
    label = subprocess.check_output(
        ["findmnt", "-no", "LABEL", "-T", str(RUNS)], text=True
    ).strip()
    if label != "SSD1":
        raise ValueError(f"{RUNS.resolve()} is on {label!r}, expected SSD1")
    print(
        f"{RUNS.resolve()} on {label}; free {shutil.disk_usage(RUNS).free / 2**30:.1f} GiB",
        flush=True,
    )
    subprocess.run(["nvidia-smi"], check=True)
    import torch

    if not torch.cuda.is_available():
        raise ValueError(f"CUDA unavailable in {sys.executable}")


def prepare(folder):
    # Pin the implementation independently of the legacy Stage 1 code in vendor/.
    source = folder / "upstream"
    source.mkdir()
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            [
                "git",
                "-C",
                str(AUDIT / "LaWAM"),
                "archive",
                SOURCE,
                "latent_action_model",
                "starVLA",
            ],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        subprocess.run(
            ["tar", "-xf", "-", "-C", str(source)], stdin=archive, check=True
        )
    decoder = source / "latent_action_model/core/utils/lam_decoder.py"
    original = "s_pred = torch.tanh(self.state_decoder(fused, emb))"
    code = decoder.read_text()
    if code.count(original) != 1:
        raise ValueError("Pinned state decoder differs from the training contract")
    decoder.write_text(
        code.replace(original, original.replace("torch.tanh", "2 * torch.tanh"))
    )
    return source


def runtime_env(source):
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            map(str, (source, ROOT, AUDIT / "python-deps", AUDIT / "pytorch3d"))
        ),
        HF_HOME=str(AUDIT / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LAM_ENABLE_MANUAL_WANDB="0",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        LAM_WORKER_OMP_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )


def launch(args):
    import yaml

    folder = run_path(args.name)
    check()
    template = (
        getattr(args, "config", None) or ROOT / "configs/stage1_difference_30k.yaml"
    )
    resume = args.command == "resume"
    if resume:
        parent = run_path(args.source)
        checkpoint = parent / "checkpoints/last.ckpt"
        configs = sorted(
            parent.glob("metrics/version_*/config.yaml"),
            key=lambda p: int(p.parent.name.split("_")[-1]),
        )
        if not checkpoint.is_file() or not configs:
            raise ValueError(
                f"Missing checkpoint or resolved Lightning config in {parent}"
            )
        template = configs[-1]
    cfg = yaml.safe_load(template.read_text())
    # Atomic reservation prevents two invocations from sharing a destination.
    folder.mkdir()
    source = prepare(folder)
    trainer = cfg["trainer"]
    trainer["default_root_dir"] = str(folder)
    for logger in trainer["logger"]:
        logger["init_args"]["save_dir"] = str(folder)
    for callback in trainer["callbacks"]:
        if callback["class_path"].endswith(".ModelCheckpoint"):
            callback["init_args"]["dirpath"] = str(folder / "checkpoints")
    if getattr(args, "max_steps", None) is not None:
        trainer["max_steps"] = args.max_steps
    if args.command == "preflight":
        trainer.update(max_steps=3, val_check_interval=3, limit_val_batches=2)
    if args.command == "train" and args.seed is not None:
        cfg["seed_everything"] = args.seed
    config = folder / "training.yaml"
    # Resolved Lightning configs include CLI-only options; supply resume explicitly.
    cfg.pop("ckpt_path", None)
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = runtime_env(source)
    cli = [
        sys.executable,
        "-m",
        "research.run_full_difference",
        "fit",
        "--config",
        str(config),
    ]
    if resume:
        cli += ["--ckpt_path", str(checkpoint)]
    print(f"Run: {folder.resolve()}", flush=True)
    os.chdir(ROOT)
    os.execve(sys.executable, cli, env)


def legacy_fit(options):
    """Preserve the historical Lightning flags with the same pinned model source."""
    with tempfile.TemporaryDirectory(prefix="difference-stage1-") as temp:
        source = prepare(Path(temp))
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "research.run_full_difference",
                "fit",
                "--config",
                str(ROOT / "configs/stage1_difference_30k.yaml"),
                *options,
            ],
            cwd=ROOT,
            env=runtime_env(source),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check SSD1, data, and available GPU")
    sub.add_parser("status", help="Read the latest recorded progress").add_argument(
        "name", nargs="?", default=CURRENT
    )
    for cmd in ("preflight", "train", "resume"):
        p = sub.add_parser(cmd)
        p.add_argument("name", help="New output directory name under SSD1 runs")
        if cmd != "resume":
            p.add_argument(
                "--config", type=Path, help="Stage 1 YAML (defaults to the 30k config)"
            )
        if cmd == "resume":
            p.add_argument(
                "--max-steps",
                type=int,
                help="Total updates including the restored step",
            )
            p.add_argument("--from-run", dest="source", required=True)
        if cmd == "train":
            p.add_argument("--seed", type=int, help="Override the YAML seed")
    args = parser.parse_args()
    if args.command == "status":
        status(args.name)
    elif args.command == "check":
        check()
    else:
        launch(args)


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--legacy-fit"]:
            sys.exit(legacy_fit(sys.argv[2:]))
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
