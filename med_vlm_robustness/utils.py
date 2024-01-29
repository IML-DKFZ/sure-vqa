from argparse import ArgumentParser

from omegaconf import OmegaConf


def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="med_vlm_robustness/med_vlm_robustness/config/training_defaults.yaml",
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