
import argparse
import yaml
from types import SimpleNamespace
import descriptors
import types

from experiment import Experiment

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in d.items()})
    return d

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return dict_to_namespace(config_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="mace_config.yaml", help='path to YAML config file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(f"configs/{args.config}")

    if config.mode == "descriptor":
        descriptor_config = config.descriptor
        params = getattr(descriptor_config, "init_params", {}) 
        params = vars(params) if isinstance(params, types.SimpleNamespace) else params
        descriptor = getattr(descriptors, descriptor_config.name)(**(params))
        descriptor.generate_descriptors(**vars(descriptor_config.generate_descriptors_params))
    
    elif config.mode == "experiments":
        for experiment_name, experiment in vars(config.experiments).items():
            print(f"Running {experiment_name}")
            exp = Experiment(**vars(experiment))
            exp.run()
    
    else:
        raise ValueError("What is this mode?!")

if __name__ == "__main__":
    main()
