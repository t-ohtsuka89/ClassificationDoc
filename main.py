import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import models
from datamodule import MyDataModule
from utils.logging import set_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="設定ファイル(.yaml)")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = set_logger(config["log"]["filename"])
    logger.info(config)

    seed = config["seed"]
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore

    label_dir = config["dataset"]["label_dir"]

    # パラメータの設定
    PADDING_IDX = 0

    dm = MyDataModule(
        config["dataset"]["text_dir"],
        label_dir,
        batch_size=config["train"]["batch_size"],
        seed=seed,
        padding_idx=PADDING_IDX,
    )

    dm.setup(stage="fit")
    vocab_size = dm.vocab_size
    output_size = dm.output_size

    model: pl.LightningModule = getattr(models, "CNN")(
        vocab_size=vocab_size,
        output_size=output_size,
        **config["model"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("models", "checkpoints"),
        filename="model_best",
        monitor="val_f1",
        verbose=False,
        save_last=False,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
    )
    early_stopping_callback = EarlyStopping(**config["early_stopping"])

    trainer = Trainer(
        gpus=1,
        max_epochs=config["train"]["n_epoch"],
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    logger.info("Training...")
    trainer.fit(model, datamodule=dm)
    logger.info("Testing...")
    trainer.test(datamodule=dm)
    logger.info("Done.")


if __name__ == "__main__":
    main(get_args())
