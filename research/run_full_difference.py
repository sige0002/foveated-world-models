import json
from research.difference_cli import SPLIT
from lightning.pytorch.cli import LightningCLI
from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
from research.full_difference import DifferenceLAM
from research.train_robotwin_stage1 import install_episode_split

if __name__ == "__main__":
    install_episode_split(json.loads(SPLIT.read_text())["ids"])
    LightningCLI(
        DifferenceLAM,
        LeRobotDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )
