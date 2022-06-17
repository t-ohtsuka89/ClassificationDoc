import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import models
from datamodule import BertDataModule, MyDataModule
from special_tokens import SpecialToken
from utils.line import send_line_notify
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
    PADDING_IDX = SpecialToken.PAD

    if config["method"] == "Bert":
        dm = BertDataModule(
            model_name=config["model"]["model_name"],
            text_dir=config["dataset"]["text_dir"],
            label_dir=label_dir,
            batch_size=config["dataset"]["batch_size"],
            seed=seed,
            padding_idx=PADDING_IDX,
        )
    else:
        dm = MyDataModule(
            config["dataset"]["text_dir"],
            label_dir,
            batch_size=config["dataset"]["batch_size"],
            seed=seed,
            add_special_token=config.get("add_special_token", False),
            padding_idx=PADDING_IDX,
        )

    dm.setup(stage="fit")
    vocab_size = dm.vocab_size
    output_size = dm.output_size

    model: pl.LightningModule = getattr(models, config["method"])(
        vocab_size=vocab_size,
        output_size=output_size,
        padding_idx=PADDING_IDX,
        **config["model"],
    )

    callbacks = []

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
    early_stopping_callback = EarlyStopping(
        check_on_train_epoch_end=config["early_stopping"]["monitor"] == "val_f1",
        **config["early_stopping"],
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(early_stopping_callback)

    trainer = Trainer(
        gpus=1,
        callbacks=callbacks,
        **config["trainer"],
    )

    if config.get("tuning_lr", False):
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        assert lr_finder is not None
        new_lr = lr_finder.suggestion()
        model.hparams["learning_rate"] = new_lr

    logger.info("Training...")
    trainer.fit(model, datamodule=dm)
    logger.info("Testing...")
    trainer.test(datamodule=dm)
    logger.info("Done.")


if __name__ == "__main__":
    main(get_args())
    send_line_notify("学習が終了しました.")
