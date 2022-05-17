import argparse
import os
import random

import numpy as np
import torch
import torch.nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datamodule import MyDataModule
from models.cnn import CNN
from utils.logging import set_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_texts_dir", default="./data/texts")
    parser.add_argument("--processed_label1_dir", default="./data/label_level1")
    parser.add_argument("--processed_label2_dir", default="./data/label_level2")
    parser.add_argument("--label", choices=["label1", "label2"], default="label2")

    parser.add_argument("--log_file", default="./train.log")
    opt = parser.parse_args()

    logger = set_logger(opt.log_file)
    logger.info(opt)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

    label_dir: str = opt.processed_label1_dir if opt.label == "label1" else opt.processed_label2_dir

    # パラメータの設定
    EMB_SIZE = 256
    PADDING_IDX = 0
    OUT_CHANNELS = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    DROP_RATE = 0.1

    dm = MyDataModule(
        opt.processed_texts_dir,
        label_dir,
        batch_size=BATCH_SIZE,
        seed=seed,
        padding_idx=PADDING_IDX,
    )

    dm.setup(stage="fit")
    VOCAB_SIZE = dm.vocab_size
    OUTPUT_SIZE = dm.output_size

    model = CNN(
        vocab_size=VOCAB_SIZE,
        output_size=OUTPUT_SIZE,
        emb_size=EMB_SIZE,
        out_channels=OUT_CHANNELS,
        drop_rate=DROP_RATE,
        padding_idx=PADDING_IDX,
        learning_rate=LEARNING_RATE,
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

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    logger.info("Training...")
    trainer.fit(model, datamodule=dm)
    logger.info("Testing...")
    trainer.test(datamodule=dm)
    logger.info("Done.")


if __name__ == "__main__":
    main()
