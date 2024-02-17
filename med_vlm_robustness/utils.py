import os
import random
from argparse import ArgumentParser

from omegaconf import OmegaConf

import numpy as np
import torch

def set_seed(seed):
    print(f"SETTING GLOBAL SEED TO {seed}")
    #pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="med_vlm_robustness/med_vlm_robustness/config/training_defaults.yaml",
        help="Config file to use",
    )
    parser.add_argument("--overrides", nargs='*', help="Specify key-value pairs to override")
    args = parser.parse_args()

    yaml_config = OmegaConf.load(args.config)

    # Merge YAML config with command line overrides
    config = OmegaConf.merge(yaml_config, OmegaConf.from_cli())

    # Apply dynamic overrides specified in the command line
    if args.overrides:
        for override in args.overrides:
            key, val = override.split('=')
            OmegaConf.update(config, key=key, value=val)

    if "model_name" in config and not config.model_name:
        config.model_name = ""

    return config