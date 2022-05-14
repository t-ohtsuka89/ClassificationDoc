import argparse
import glob
import os
import string
import time
from collections import defaultdict

import MeCab
import numpy
import regex
import torch
import torch.nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


class CNN(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        output_size,
        out_channels,
        drop_rate,
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.n_grams = [2, 3, 4]
        for i in range(len(self.n_grams)):
            conv = torch.nn.Conv2d(
                1,
                out_channels,
                (self.n_grams[i], emb_size),
                padding=(i, 0),
            )
            setattr(self, f"conv_{i}", conv)
        self.drop = torch.nn.Dropout(drop_rate)
        self.output = torch.nn.Linear(out_channels * 3, output_size)

    def get_conv(self, i: int):
        return getattr(self, f"conv_{i}")

    def forward(self, x: Tensor):
        emb: Tensor = self.emb(x)

        conv_results: list[Tensor] = []
        for i in range(len(self.n_grams)):
            conv_x: Tensor = self.get_conv(i)(emb.unsqueeze(1))
            conv_x = F.relu(conv_x.squeeze(3))
            conv_x = F.max_pool1d(conv_x, conv_x.size()[2])
            conv_x = conv_x.squeeze(2)
            conv_results.append(conv_x)

        out = torch.cat(conv_results, dim=1)
        out = self.output(self.drop(out))
        return out


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


def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None, OUTPUT_SIZE=None):
    """損失・正解率を計算"""
    PADDING_IDX = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Padsequence(PADDING_IDX))
    loss = 0.0
    total = 0
    acc_sum = 0.0
    f1_sum = 0.0
    all_pred = numpy.empty([0, OUTPUT_SIZE])
    all_labels = numpy.empty([0, OUTPUT_SIZE])
    with torch.no_grad():
        for batch in dataloader:
            batch: dict[str, Tensor]
            # デバイスの指定
            inputs = batch["input"].to(device)
            labels = batch["labels"].to(device)
            # print(labels)
            # 順伝播
            outputs = model(inputs)

            # 損失計算
            if criterion != None:
                loss += criterion(outputs, labels).item()

            # 正解率計算
            outputs_pred = torch.sigmoid(outputs)
            pred_round = torch.round(outputs_pred)

            y_true = labels.cpu().data.numpy()
            y_pred = pred_round.cpu().data.numpy()

            all_pred = numpy.append(all_pred, y_pred, axis=0)
            all_labels = numpy.append(all_labels, y_true, axis=0)

        all_labels_int = all_labels.astype("int64")
        all_pred_int = all_pred.astype("int64")
        f1_micro = f1_score(all_labels_int, all_pred_int, average="micro")
        accuracy = accuracy_score(all_labels_int, all_pred_int)
    return loss / len(dataset), accuracy, f1_micro  # acc / total, f1_sum / total


def train_model(
    dataset_train,
    dataset_valid,
    batch_size,
    model,
    criterion,
    optimizer,
    num_epochs,
    collate_fn=None,
    device=None,
    OUTPUT_SIZE=None,
):

    # デバイスの指定
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # スケジューラの設定
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 開始時刻の記録
        s_time = time.time()
        cnt = 1
        # 訓練モードに設定
        model.train()
        for batch in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = batch["input"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            # print(cnt)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cnt = cnt + 1
        # 評価モードに設定
        model.eval()

        # 損失と正解率の算出
        # loss_train, acc_train, _ = calculate_loss_and_accuracy(model, dataset_train, device, criterion=criterion, OUTPUT_SIZE=OUTPUT_SIZE)
        loss_valid, acc_valid, f1_valid = calculate_loss_and_accuracy(
            model, dataset_valid, device, criterion=criterion, OUTPUT_SIZE=OUTPUT_SIZE
        )
        # log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid, f1_valid])

        # チェックポイントの保存
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"checkpoint{epoch + 1}.pt",
        )

        # 終了時刻の記録
        e_time = time.time()

        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, f1_valid: {f1_valid:.4f}, {(e_time - s_time):.4f}sec"
        )

        # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
        if (
            epoch > 2
            and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]
        ):
            break

        # スケジューラを1ステップ進める
        scheduler.step()

    return {"train": log_train, "valid": log_valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_texts_dir", default="./data/texts")
    parser.add_argument("--processed_label1_dir", default="./data/label_level1")
    parser.add_argument("--processed_label2_dir", default="./data/label_level2")
    parser.add_argument("--label", choices=["label1", "label2"], default="label2")
    parser.add_argument("--method", choices=["LinearSVC", "SVC"])
    parser.add_argument("--show_report", action="store_true", help="Show evaluation details")

    parser.add_argument("--log_file", default="./train_log.log")
    opt = parser.parse_args()

    logger = set_logger(opt.log_file)
    logger.info(opt)

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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

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
        text_list, label_list, test_size=0.2, shuffle=True, random_state=123
    )
    X_val, X_test, y_val, y_test = train_test_split(
        val_test_text, val_test_label, test_size=0.5, shuffle=True, random_state=123
    )

    dataset_train = CreateDataset(X_train, y_train)
    dataset_valid = CreateDataset(X_val, y_val)
    dataset_test = CreateDataset(X_test, y_test)

    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 2  # 辞書のID数 + パディングID
    EMB_SIZE = 100
    PADDING_IDX = 0
    OUTPUT_SIZE = len(mlb.classes_)  # ラベルの総種類数
    OUT_CHANNELS = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 80
    NUM_EPOCHS = 100
    DROP_RATE = 0.1

    print(OUTPUT_SIZE)

    model = CNN(
        VOCAB_SIZE,
        EMB_SIZE,
        OUTPUT_SIZE,
        OUT_CHANNELS,
        DROP_RATE,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Training...")
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
        OUTPUT_SIZE=OUTPUT_SIZE,
    )
    logger.info("Testing...")
    loss_test, acc_test, f1_test = calculate_loss_and_accuracy(
        model, dataset_test, device, criterion=criterion, OUTPUT_SIZE=OUTPUT_SIZE
    )
    logger.info("Done.")

    logger.info(f"loss_test:{loss_test}")
    logger.info(f"accuracy score:{acc_test}")
    logger.info(f"micro-f1 score:{f1_test}")


if __name__ == "__main__":
    main()
