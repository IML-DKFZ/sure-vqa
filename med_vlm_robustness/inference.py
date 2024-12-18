import os
import time
from pathlib import Path

import torch.cuda

from datamodule import get_datamodule
from model import LLaVA_Med
from pytorch_lightning import Trainer
from utils import get_config, set_seed


def main(cfg):
    if "seed" in cfg:
        cfg.seed = int(cfg.seed)
        set_seed(cfg.seed)

    if not cfg.get("corruption", False):
        cfg.corruption = False
        cfg.corruption_probabilities = None
        cfg.corruption_strength = None

    dm, split_file_name = get_datamodule(data_dir=Path(cfg.data_dir),
                       ood_value=cfg.ood_value, test_folder_name=cfg.test_folder_name,
                       train_folder_name=cfg.train_folder_name, val_folder_name=cfg.val_folder_name, 
                       dataset_name=cfg.dataset, split=cfg.split, data_shift=cfg.data_shift, 
                       batch_size=cfg.batch_size, num_workers=cfg.num_workers, mod=cfg.mod, no_image=cfg.no_image, 
                       corruption=cfg.corruption, corruption_probabilities=cfg.corruption_probabilities, corruption_strength = cfg.corruption_strength)

    dm.setup()

    split_file_train = split_file_name.replace(cfg.mod, 'train').replace('ood', 'iid')

    # in case the train split doesnt use sample dataset but inference does
    if 'sample' not in cfg.train_split:
        split_file_train = split_file_train.replace('sample_iid', 'iid')

    if not cfg.train_no_image and cfg.no_image: # When the model is trained with images but inference is without images
        split_file_train = split_file_train.split('_no_image')[0]
    
    if cfg.train_no_image and not cfg.no_image: # When the model is trained without images but inference is with images
        split_file_train = split_file_train + '_no_image'

    print(f"Split file train: {split_file_train}")
    if "model_name" not in cfg:
        cfg["model_name"] = f"llava-{split_file_train}-finetune_{cfg.model_type}"
    if "hyperparams_model_name" in cfg and cfg.hyperparams_model_name is not None:
        cfg.model_name = f"{cfg.model_name}_{cfg.hyperparams_model_name}"

    if "model_path" not in cfg:
        if not cfg.get("extend_experiment_dir", False):
            cfg["model_path"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/{cfg.model_name}"
        else:
            medical_str = "medical" if cfg.get("is_medical", True) else "non_medical"
            cfg["model_path"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{medical_str}/{cfg.dataset}/{cfg.model_type}/{cfg.model_name}"
    print(f"Model path: {cfg.model_path}")

    if cfg.corruption:
        split_file_name = split_file_name + '_corruption' 
        strength = cfg.corruption_strength['blur']
        split_file_name = split_file_name + '_' + str(strength) 

    # print('CORRUPTION: ', cfg.corruption)
    # print(str(cfg.corruption_strength))

    if "output_file" not in cfg:
        if cfg.model_type != "pretrained":
            cfg["output_file"] = f"{cfg.model_path}/eval/{split_file_name}/test_results.json"
        else:
            if not cfg.get("extend_experiment_dir", False):
                cfg["output_file"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/eval/{split_file_name}/test_results.json"
            else:
                medical_str = "medical" if cfg.get("is_medical", True) else "non_medical"
                cfg[
                    "output_file"] = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{medical_str}/{cfg.dataset}/{cfg.model_type}/eval/{split_file_name}/test_results.json"
    llava = LLaVA_Med(cfg)

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(llava, datamodule=dm)

    #os.system(f'touch {cfg["model_path"]}.done')

if __name__ == "__main__":
    config = get_config()
    main(cfg=config)
