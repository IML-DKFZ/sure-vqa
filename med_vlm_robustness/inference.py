from pathlib import Path

import torch.cuda

from datamodule import get_datamodule
from model import LLaVA_Med
from pytorch_lightning import Trainer
from utils import get_config


def main(cfg):
    llava = LLaVA_Med(cfg)
    dm = get_datamodule(Path(cfg.data_dir), cfg.split_file, batch_size=1, num_workers=cfg.num_workers)
    dm.setup()

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

if __name__ == "__main__":
    config = get_config()
    main(cfg=config)
