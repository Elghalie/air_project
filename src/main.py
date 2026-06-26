import sys
import os
import matplotlib

# Backend non interactif
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from src.utils import load_config, set_seed


def main():
    # Chargement du fichier YAML
    config = load_config("configs/config.yaml")

    # Fixation de la graine aléatoire
    set_seed(config["project"]["random_seed"])

    print("[RUN] Execution du pipeline scientifique...")
    print("Configuration chargée :")
    print(config)


if __name__ == "__main__":
    main()