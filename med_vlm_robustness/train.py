from argparse import ArgumentParser
from llava.train.train import *
from omegaconf import OmegaConf

from med_vlm_robustness.datamodule import get_json_filename


def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../config/slake_lora.yaml",
        help="Config file to use",
    )
    parser.add_argument("--overrides", nargs='+', help="Specify key-value pairs to override")
    args = parser.parse_args()

    yaml_config = OmegaConf.load(args.config)

    # Merge YAML config with command line overrides
    config = OmegaConf.merge(yaml_config, OmegaConf.from_cli())

    # Apply dynamic overrides specified in the command line
    if args.overrides:
        for override in args.overrides:
            key, val = override.split('=')
            OmegaConf.update(config, key=key, value=val)

    return config


def main(cfg):
    data_root_dir = "/nvme/VLMRobustness"
    dataset = "Slake"
    split_file = "slake_train_all"
    output_dir = f"{data_root_dir}/{dataset}/Experiments/LoRA"

    model_args = ModelArguments(cfg.model_args)
    data_args = DataArguments(cfg.data_args)
    training_args = TrainingArguments(output_dir=output_dir)
    # TODO: Workaround for now because derived dataclasses cannot be instantiated with base class fields
    for key, value in cfg.training_args.items():
        setattr(training_args, key, value)

    # get dataset json
    data_args.data_path = get_json_filename(split_file)
    data_args.image_folder = f"{data_root_dir}/{dataset}"

    train(data_args=data_args, model_args=model_args, training_args=training_args)


if __name__=="__main__":
    config = get_config()
    main(cfg=config)
