# Simple file to print the config in Hydra

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path = "../configs", config_name = "dummyconfig", version_base = None)
def process_data(config: DictConfig):
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    process_data()