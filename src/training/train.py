from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, MLFlowLogger


log = logging.Logger(__name__)

def train(cfg: DictConfig):

# Seeding ensures reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers = True)

    log.info(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating Lightning Module <{cfg.model._target_}>")
    model : pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(f"Instantiating Loggers ")
    # logger : List[Logger] = (cfg.get("logger"))

    # log.info(f"Instantiating Callbacks...")
    # callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)




@hydra.main(config_path = "configs", config_name = "train", version_base = None)
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()