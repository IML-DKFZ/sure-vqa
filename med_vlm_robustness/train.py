from pathlib import Path

from llava.train.train import *

from med_vlm_robustness.datamodule import get_json_filename
from med_vlm_robustness.utils import get_config


def main(cfg):
    model_args = ModelArguments(**cfg.model_args)
    data_args = DataArguments(**cfg.data_args)
    training_args = TrainingArguments(output_dir=cfg.output_dir)
    # TODO: Workaround for now because derived dataclasses cannot be instantiated with base class fields
    for key, value in cfg.training_args.items():
        setattr(training_args, key, value)

    # get dataset json
    data_args.data_path = get_json_filename(Path(cfg.data_dir), cfg.split_file)

    train(data_args=data_args, model_args=model_args, training_args=training_args)


if __name__=="__main__":
    config = get_config()
    main(cfg=config)
