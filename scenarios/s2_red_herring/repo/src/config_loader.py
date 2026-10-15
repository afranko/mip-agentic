import yaml


def load_config(path):
    """Load service settings from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)
