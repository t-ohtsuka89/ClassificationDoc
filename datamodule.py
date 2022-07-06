import glob
import os
import string
from collections import defaultdict
from typing import Optional

import MeCab
import pytorch_lightning as pl
import regex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from collate_fn import Padsequence
from dataset import CreateBertDataset, CreateDataset
from special_tokens import SpecialToken


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_dir: str,
        label_dir: str,
        batch_size: int,
        seed: int,
        add_special_token: bool = False,
        n_truncation: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.collate_fn = Padsequence(SpecialToken.PAD)
        self.special_token_num = len(SpecialToken)
        assert len(SpecialToken) == 5

    def setup(self, stage: str | None):
        tagger = MeCab.Tagger("-Owakati")
        text_list, label_list = self.make_dataset(self.hparams["label_dir"], self.hparams["text_dir"])
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

        text_list = [self.tokenizer(line, word2id) for line in text_list]
        if self.hparams["n_truncation"] is not None:
            text_list = [self.truncation(ids, self.hparams["n_truncation"]) for ids in text_list]

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
        self.max_len = max(map(len, text_list))

    def train_dataloader(self):
        """
        Returns:
            DataLoader: dataloader for training
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """
        Returns:
            DataLoader: dataloader for validation
        """
        return DataLoader(self.dataset_valid, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

    def tokenizer(self, text: str, word2id: dict[str, int]) -> list[int]:
        """
        Args:
            text: str
            word2id: dict[str, int]
        Returns:
            list[int]: tokenized text
        """

        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(table)
        words = text.split()
        ids = [word2id[word] if word in word2id else SpecialToken.UNK for word in words]
        if self.hparams["add_special_token"]:
            # add special token
            ids = [SpecialToken.BOS] + ids + [SpecialToken.EOS]
        return ids

    def make_dataset(self, labels_dir: str, texts_dir: str):
        """
        Args:
            labels_dir: path to labels
            texts_dir: path to texts
        Returns:
            list: text list
            list: label list
        """
        text_list: list[str] = []
        label_list: list[list[str]] = []

        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        file_ids = [self.get_fileid(filepath) for filepath in text_files]
        for file_id in file_ids:
            try:
                paths = glob.glob(os.path.join(texts_dir, file_id + "_k_?.txt"))
                assert len(paths) == 1
                with open(paths[0], "r") as text_file, open(
                    os.path.join(labels_dir, file_id + "_k_l.txt"), "r"
                ) as label_file:
                    text = text_file.read().strip()
                    label = label_file.read().strip().split("\n")
                    if "" in label:
                        continue
                    text_list.append(text)
                    label_list.append(label)
            except FileNotFoundError:
                continue

        return text_list, label_list

    @staticmethod
    def get_fileid(filepath: str) -> str:
        """
        Args:
            filepath: path to file
        Returns:
            str: file id
        """
        file_basename = os.path.basename(filepath)
        p = regex.match(r"(.*)_k_(s|y).txt", file_basename)
        assert p is not None
        fileid = p.groups()[0]
        return fileid

    @staticmethod
    def truncation(seq: list, max_seq_len=512):
        """
        Args:
            seq: list of ids
            max_seq_len: max length of sequence
        Returns:
            list: truncated sequence
        """
        if len(seq) > max_seq_len:
            return seq[:max_seq_len]
        else:
            return seq


class TransformersDataModule(pl.LightningDataModule):
    """
    DataModule for Transformers
    """

    def __init__(
        self,
        model_name: str,
        text_dir: str,
        label_dir: str,
        batch_size: int,
        seed: int,
        add_special_token: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None):
        text_list, label_list = self.make_dataset(self.hparams["label_dir"], self.hparams["text_dir"])
        tokenizer = AutoTokenizer.from_pretrained(self.hparams["model_name"], do_lower_case=False)
        assert isinstance(tokenizer, PreTrainedTokenizerFast | PreTrainedTokenizer)
        mlb = MultiLabelBinarizer()
        mlb.fit(label_list)
        label_list = mlb.transform(label_list)
        inputs: BatchEncoding = tokenizer.batch_encode_plus(
            text_list, padding="max_length", max_length=512, return_tensors="np", truncation=True
        )

        input_ids_train, val_test_input_ids, mask_train, val_test_mask, y_train, val_test_label = train_test_split(
            inputs["input_ids"],
            inputs["attention_mask"],
            label_list,
            test_size=0.2,
            shuffle=True,
            random_state=self.hparams["seed"],
        )
        input_ids_val, input_ids_test, mask_val, mask_test, y_val, y_test = train_test_split(
            val_test_input_ids,
            val_test_mask,
            val_test_label,
            test_size=0.5,
            shuffle=True,
            random_state=self.hparams["seed"],
        )
        self.dataset_train = CreateBertDataset(input_ids_train, y_train, mask_train)
        self.dataset_valid = CreateBertDataset(input_ids_val, y_val, mask_val)
        self.dataset_test = CreateBertDataset(input_ids_test, y_test, mask_test)
        self.vocab_size = tokenizer.vocab_size
        self.output_size = len(mlb.classes_)
        self.max_len = max(map(len, text_list))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False)

    def make_dataset(self, labels_dir: str, texts_dir: str):
        """
        Args:
            labels_dir: path to labels
            texts_dir: path to texts
        Returns:
            list: text list
            list: label list
        """
        text_list: list[str] = []
        label_list: list[list[str]] = []

        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        file_ids = [self.get_fileid(filepath) for filepath in text_files]
        for file_id in file_ids:
            try:
                paths = glob.glob(os.path.join(texts_dir, file_id + "_k_?.txt"))
                assert len(paths) == 1
                with open(paths[0], "r") as text_file, open(
                    os.path.join(labels_dir, file_id + "_k_l.txt"), "r"
                ) as label_file:
                    text = text_file.read().strip()
                    label = label_file.read().strip().split("\n")
                    if "" in label:
                        continue
                    text_list.append(text)
                    label_list.append(label)
            except FileNotFoundError:
                continue

        return text_list, label_list

    @staticmethod
    def get_fileid(filepath: str) -> str:
        file_basename = os.path.basename(filepath)
        p = regex.match(r"(.*)_k_(s|y).txt", file_basename)
        assert p is not None
        fileid = p.groups()[0]
        return fileid
