import yaml
import random
import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)