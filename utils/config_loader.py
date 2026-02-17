import yaml

def load_config(path="config/base_config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
