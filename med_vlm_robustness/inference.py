from pathlib import Path

import torch.cuda

from med_vlm_robustness.datamodule import get_datamodule
from med_vlm_robustness.model import LLaVA_Med
from pytorch_lightning import Trainer
from med_vlm_robustness.utils import get_config


def main(cfg):
    llava = LLaVA_Med(cfg)
    dm = get_datamodule(Path(cfg.data_dir), cfg.split_file, batch_size=1)
    dm.setup()

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

if __name__ == "__main__":
    config = get_config()
    main(cfg=config)
