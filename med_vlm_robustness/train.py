import os
from pathlib import Path
from llava.train.train import *
from datamodule import get_json_filename
from utils import get_config


def main(cfg):
    model_args = ModelArguments(**cfg.model_args)
    data_args = DataArguments(**cfg.data_args)

    # get dataset json
    data_args.data_path, split_file_name = get_json_filename(Path(cfg.data_dir),
                                            ood_value=cfg.ood_value,test_folder_name=cfg.test_folder_name,
                                            train_folder_name=cfg.train_folder_name,val_folder_name=cfg.val_folder_name, 
                                            dataset_name=cfg.dataset, split=cfg.train_split, 
                                            data_shift=cfg.data_shift, mod="train") # mod = train / val / test
    model_name = f"llava-{split_file_name}-finetune_{cfg.model_type}"
    cfg.output_dir = Path(os.getenv("EXPERIMENT_ROOT_DIR")) / cfg.dataset / cfg.model_type / model_name

    if cfg.hyperparams_model_name is not None:
        cfg.output_dir = f"{cfg.output_dir}_{cfg.hyperparams_model_name}"

    training_args = TrainingArguments(output_dir=cfg.output_dir, run_name=model_name)
    # TODO: Workaround for now because derived dataclasses cannot be instantiated with base class fields
    for key, value in cfg.training_args.items():
        setattr(training_args, key, value)

    train(data_args=data_args, model_args=model_args, training_args=training_args)


if __name__=="__main__":
    config = get_config()
    main(cfg=config)
