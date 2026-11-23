import json
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
from research.full_difference import DifferenceLAM
from research.train_robotwin_stage1 import install_episode_split

if __name__ == '__main__':
    split = Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114/kl_mse/episode_split.json')
    install_episode_split(json.loads(split.read_text())['ids'])
    LightningCLI(DifferenceLAM, LeRobotDataModule, seed_everything_default=42,
                 save_config_kwargs={'overwrite': True})
