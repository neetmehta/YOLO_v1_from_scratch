import yaml





def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)