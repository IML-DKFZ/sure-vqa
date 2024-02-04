from pathlib import Path

import torch.cuda

from datamodule import get_datamodule
from model import LLaVA_Med
from pytorch_lightning import Trainer
from utils import get_config


def main(cfg):
    llava = LLaVA_Med(cfg)

    dm = get_datamodule(data_dir=Path(cfg.data_dir), output_file_name=cfg.split_file, 
                       ood_value=cfg.ood_value, test_folder_name=cfg.test_folder_name,
                       train_folder_name=cfg.train_folder_name, val_folder_name=cfg.val_folder_name, 
                       dataset_name=cfg.dataset, split=cfg.split, data_shift=cfg.data_shift, 
                       batch_size=cfg.batch_size, num_workers=cfg.num_workers, mod="test",)
    dm.setup()

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

if __name__ == "__main__":
    config = get_config()
    main(cfg=config)
