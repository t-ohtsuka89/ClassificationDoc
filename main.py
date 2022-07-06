import argparse
from logging import DEBUG
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

import models
from datamodule import MyDataModule, TransformersDataModule
from special_tokens import SpecialToken
from utils.line import send_line_notify
from utils.logging import set_logger


def build_config(args: argparse.Namespace) -> dict:
    """
    Build config file.
    """
    with open(args.config, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def get_callbacks(config: dict) -> list:
    """
    Get callbacks.
    """
    callbacks = []
    if config.get("save_top_k", 1) > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["save_dir"],
                save_top_k=config["save_top_k"],
                monitor="val_f1",
                mode="max",
            )
        )
    if config["early_stopping"]["patience"] > 0:
        callbacks.append(
            EarlyStopping(
                verbose=True,
                check_on_train_epoch_end=config["early_stopping"]["monitor"] == "val_f1",
                **config["early_stopping"],
            )
        )
    swa_lrs = config.get("swa_lrs", None)
    if swa_lrs is not None:
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
    return callbacks


def get_args() -> argparse.Namespace:
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    main関数
    """

    # Load config file
    config = build_config(args)

    # Set loggerc
    set_logger(config["log_path"], config.get("log_level", DEBUG))

    # Set seed
    pl.seed_everything(config["seed"])
    torch.backends.cudnn.deterministic = True  # type: ignore

    # Create data module
    if config["dataset"]["data_module"] == "MyDataModule":
        data_module = MyDataModule(
            text_dir=config["dataset"]["text_dir"],
            label_dir=config["dataset"]["label_dir"],
            batch_size=config["dataset"]["batch_size"],
            seed=config["seed"],
            add_special_token=config.get("add_special_token", False),
            n_truncation=config.get("n_truncation", None),
        )
    elif config["dataset"]["data_module"] == "TransformersDataModule":
        data_module = TransformersDataModule(
            model_name=config["model"]["model_name"],
            text_dir=config["dataset"]["text_dir"],
            label_dir=config["dataset"]["label_dir"],
            batch_size=config["dataset"]["batch_size"],
            seed=config["seed"],
            add_special_token=config.get("add_special_token", False),
        )
    else:
        raise ValueError("Unknown data module.")

    # Create model
    data_module.setup(stage="fit")
    model: pl.LightningModule = getattr(models, config["method"])(
        vocab_size=data_module.vocab_size,
        output_size=data_module.output_size,
        padding_idx=SpecialToken.PAD,
        **config["model"],
    )

    # Create callbacks
    callbacks = get_callbacks(config)

    # Create trainer
    trainer = Trainer(
        callbacks=callbacks,
        **config["trainer"],
    )

    assert isinstance(config["trainer"], dict)
    if config["trainer"].get("auto_lr_find", False):
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
        assert lr_finder is not None
        new_lr = lr_finder.suggestion()
        model.hparams["learning_rate"] = new_lr

    # Train model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Send line notify
    metrics = trainer.callback_metrics
    send_line_notify(f"{config['prefix']} finished.")
    send_line_notify(f"metrics: {metrics}")
    send_line_notify(f"config: {config}")


if __name__ == "__main__":
    main(get_args())
