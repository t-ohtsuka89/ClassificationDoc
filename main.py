import argparse
import glob
import os
import random
import string
from collections import defaultdict

import MeCab
import numpy as np
import regex
import torch
import torch.nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.cnn import CNN
from utils.logging import set_logger


class FeatureVectorizer:
    def __init__(self):
        self.mecab_wakati_tagger = MeCab.Tagger("-Owakati")

    def make_feature_vector(self, corpus_list: list[str]) -> list[str]:
        return [self.mecab_wakati_tagger.parse(text) for text in tqdm(corpus_list, ncols=70)]


def make_dataset(labels_dir: str, texts_dir: str):
    text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
    text_list: list[str] = []
    label_list: list[list[str]] = []
    for text_file in text_files:
        file_basename = os.path.basename(text_file)
        p = regex.match(r"(.*)_k_s.txt", file_basename)
        assert p is not None
        file_id = p.groups()[0]

        with open(os.path.join(texts_dir, file_id + "_k_s.txt"), "r") as text_file, open(
            os.path.join(labels_dir, file_id + "_k_l.txt"), "r"
        ) as label_file:
            text = text_file.read().strip()
            label = label_file.read().strip().split("\n")
            if "" in label:
                continue
            text_list.append(text)
            label_list.append(label)

    return text_list, label_list


def tokenizer(text: str, word2id: dict[str, int], unk: int = 1):
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return [word2id.get(word, unk) for word in text.translate(table).split()]


class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return {
            "input": torch.tensor(self.X[index], dtype=torch.int64),
            "labels": torch.tensor(self.y[index], dtype=torch.float32),
        }


class Padsequence:
    """Dataloaderからミニバッチを取り出すごとに最大系列長でパディング"""

    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        inputs = [sample["input"] for sample in batch]
        labels_list = [sample["labels"] for sample in batch]
        padded_inputs = pad_sequence(inputs, batch_first=True)  # padding
        return {"input": padded_inputs.contiguous(), "labels": torch.stack(labels_list).contiguous()}


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
    text_list, label_list = make_dataset(label_dir, opt.processed_texts_dir)

    logger.info("Vectorize text...")
    vectorizer = FeatureVectorizer()
    text_list = vectorizer.make_feature_vector(text_list)
    logger.info("Done.")

    logger.info("Multi label binarize...")
    mlb = MultiLabelBinarizer()
    label_list = mlb.fit_transform(label_list)
    logger.info("Done.")

    word_count: defaultdict[str, int] = defaultdict(int)
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    for text in text_list:
        text: str
        for word in text.translate(table).split():
            word_count[word] += 1
    word_freq_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    word2id: dict[str, int] = {word: i + 2 for i, (word, cnt) in enumerate(word_freq_list) if cnt > 0}
    # IDへ変換
    text_list = [tokenizer(line, word2id) for line in tqdm(text_list)]

    X_train, val_test_text, y_train, val_test_label = train_test_split(
        text_list, label_list, test_size=0.2, shuffle=True, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        val_test_text, val_test_label, test_size=0.5, shuffle=True, random_state=seed
    )

    dataset_train = CreateDataset(X_train, y_train)
    dataset_valid = CreateDataset(X_val, y_val)
    dataset_test = CreateDataset(X_test, y_test)

    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 2  # 辞書のID数 + パディングID
    EMB_SIZE = 256
    PADDING_IDX = 0
    OUTPUT_SIZE = len(mlb.classes_)  # ラベルの総種類数
    OUT_CHANNELS = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    DROP_RATE = 0.1

    print(OUTPUT_SIZE)

    train_dataloader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Padsequence(PADDING_IDX)
    )
    val_dataloader = DataLoader(dataset_valid, batch_size=1, shuffle=False, collate_fn=Padsequence(PADDING_IDX))
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=Padsequence(PADDING_IDX))

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
        dirpath="models",
        filename="model_best",
        monitor="val_loss",
        verbose=False,
        save_last=False,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
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
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    logger.info("Testing...")
    trainer.test(dataloaders=test_dataloader)
    logger.info("Done.")


if __name__ == "__main__":
    main()
