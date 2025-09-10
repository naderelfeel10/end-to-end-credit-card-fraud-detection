import logging
import yaml
import random
import numpy as np
import os


def load_config(path="config/config.yaml"):

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/project.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)