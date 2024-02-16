import os
import time
from pathlib import Path

import torch.cuda

from datamodule import get_datamodule
from model import LLaVA_Med
from pytorch_lightning import Trainer
from utils import get_config


def main(cfg):
    llava = LLaVA_Med(cfg)

    dm, split_file_name = get_datamodule(data_dir=Path(cfg.data_dir),
                       ood_value=cfg.ood_value, test_folder_name=cfg.test_folder_name,
                       train_folder_name=cfg.train_folder_name, val_folder_name=cfg.val_folder_name, 
                       dataset_name=cfg.dataset, split=cfg.split, data_shift=cfg.data_shift, 
                       batch_size=cfg.batch_size, num_workers=cfg.num_workers, mod=cfg.mod)

    dm.setup()

    split_file_train = split_file_name.replace(cfg.mod, 'train').replace('ood', 'iid')
    if "model_path" not in cfg:
        cfg["model_path"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/llava-{split_file_train}-finetune_{cfg.model_type}"

    if "output_file" not in cfg:
        if cfg.model_type != "pretrained":
            cfg["output_file"] = f"{cfg.model_path}/eval/{split_file_name}/test_results.json"
        else:
            cfg["output_file"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/eval/{split_file_name}/test_results.json"

    llava = LLaVA_Med(cfg)

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

if __name__ == "__main__":
    config = get_config()
    main(cfg=config)
