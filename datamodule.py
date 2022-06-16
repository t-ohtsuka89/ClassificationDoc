import glob
import os
import string
from collections import defaultdict

import MeCab
import pytorch_lightning as pl
import regex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

from collate_fn import Padsequence
from dataset import CreateDataset
from special_tokens import SpecialToken


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_dir: str,
        label_dir: str,
        batch_size: int,
        seed: int,
        add_special_token: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.collate_fn = Padsequence(padding_idx)
        self.special_token_num = len(SpecialToken)
        assert len(SpecialToken) == 5

    def setup(self, stage: str | None):
        tagger = MeCab.Tagger("-Owakati")
        text_list, label_list = self.make_dataset(
            self.hparams["label_dir"], self.hparams["text_dir"]
            )
        text_list = [tagger.parse(text) for text in text_list]
        mlb = MultiLabelBinarizer()
        label_list = mlb.fit_transform(label_list)

        word_count: defaultdict[str, int] = defaultdict(int)
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        for text in text_list:
            text: str
            for word in text.translate(table).split():
                word_count[word] += 1
        word_freq_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        word2id: dict[str, int] = {
            word: i + self.special_token_num for i, (word, cnt) in enumerate(word_freq_list) if cnt > 0
        }
        # IDへ変換
        text_list = [self.tokenizer(line, word2id, SpecialToken.UNK) for line in text_list]

        X_train, val_test_text, y_train, val_test_label = train_test_split(
            text_list,
            label_list,
            test_size=0.2,
            shuffle=True,
            random_state=self.hparams["seed"],
        )
        X_val, X_test, y_val, y_test = train_test_split(
            val_test_text,
            val_test_label,
            test_size=0.5,
            shuffle=True,
            random_state=self.hparams["seed"],
        )
        self.dataset_train = CreateDataset(X_train, y_train)
        self.dataset_valid = CreateDataset(X_val, y_val)
        self.dataset_test = CreateDataset(X_test, y_test)
        self.vocab_size = len(set(word2id.values())) + self.special_token_num
        self.output_size = len(mlb.classes_)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid, batch_size=1, shuffle=False, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=1, shuffle=False, collate_fn=self.collate_fn
        )

    def tokenizer(self, text: str, word2id: dict[str, int]):
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        if self.hparams["add_special_token"]:
            l = [word2id.get(word, SpecialToken.UNK) for word in text.translate(table).split()]
            l.insert(0, SpecialToken.CLS)
            return l
        else:
            return [word2id.get(word, SpecialToken.UNK) for word in text.translate(table).split()]

    def make_dataset(self, labels_dir: str, texts_dir: str):
        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        text_list: list[str] = []
        label_list: list[list[str]] = []
        for text_file in text_files:
            file_basename = os.path.basename(text_file)
            p = regex.match(r"(.*)_k_s.txt", file_basename)
            assert p is not None
            file_id = p.groups()[0]

            with open(
                os.path.join(texts_dir, file_id + "_k_s.txt"), "r"
            ) as text_file, open(
                os.path.join(labels_dir, file_id + "_k_l.txt"), "r"
            ) as label_file:
                text = text_file.read().strip()
                label = label_file.read().strip().split("\n")
                if "" in label:
                    continue
                text_list.append(text)
                label_list.append(label)

        return text_list, label_list
