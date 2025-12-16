import os
import yaml


def config_loader(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f'Error: Config file not found at {path}')
        return
    except yaml.YAMLError as exc:
        print(f'Error parsing config file: {exc}')
        return

    for key, path in config['output'].items():
        if key.endswith('_dir'):
            os.makedirs(path, exist_ok=True)

    return config