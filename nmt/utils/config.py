import yaml


def get_config(config_path: str):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config
