import argparse
import glob
import os
import string
import time
from collections import defaultdict
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger

import MeCab
import numpy
import regex
import torch
import torch.nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FeatureVectorizer:
    def __init__(self):
        self.mecab_wakati_tagger = MeCab.Tagger("-Owakati")
        self.mecab_tagger = MeCab.Tagger()

    def make_feature_vector(self, corpus_list):
        return [self.mecab_wakati_tagger.parse(text) for text in tqdm(corpus_list, ncols=70)]


def set_logger(logfile):
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter("%(asctime)s |%(levelname)s| %(message)s"))

    file_handler = FileHandler(filename=logfile)
    file_handler.setFormatter(Formatter("%(asctime)s |%(levelname)s| %(message)s"))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def make_dataset(labels_dir, texts_dir):
    text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
    text_list = []
    label_list = []
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


class CNN(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        output_size,
        out_channels,
        kernel_heights,
        stride,
        padding,
        drop_rate,
        emb_weights=None,
    ):
        super().__init__()
        self.kernel_heights = kernel_heights
        self.Emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # self.Emb = torch.nn.Embedding.from_pretrained(emb_weights, padding_idx=0)
        self.conv1 = torch.nn.Conv2d(1, out_channels, (2, emb_size), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(1, out_channels, (3, emb_size), padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(1, out_channels, (4, emb_size), padding=(2, 0))
        self.drop = torch.nn.Dropout(drop_rate)
        self.output = torch.nn.Linear(out_channels * 3, output_size)

    def forward(self, x, x_len):
        emb = self.Emb(x)

        conv1 = self.conv1(emb.unsqueeze(1))
        conv2 = self.conv2(emb.unsqueeze(1))
        conv3 = self.conv3(emb.unsqueeze(1))

        relu1 = F.relu(conv1.squeeze(3))
        relu2 = F.relu(conv2.squeeze(3))
        relu3 = F.relu(conv3.squeeze(3))

        max_pool1 = F.max_pool1d(relu1, relu1.size()[2])
        max_pool2 = F.max_pool1d(relu2, relu2.size()[2])
        max_pool3 = F.max_pool1d(relu3, relu3.size()[2])

        out1 = max_pool1.squeeze(2)
        out2 = max_pool2.squeeze(2)
        out3 = max_pool3.squeeze(2)

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.output(self.drop(out))
        return out


class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        inputs = self.X[index]

        return {
            "inputs": torch.tensor(inputs, dtype=torch.int64),
            "labels": torch.tensor(self.y[index], dtype=torch.float64),
        }


class Padsequence:
    """Dataloaderからミニバッチを取り出すごとに最大系列長でパディング"""

    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x["inputs"].shape[0], reverse=True)
        sequences = [x["inputs"] for x in sorted_batch]
        sequences_len = [len(x["inputs"]) for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.stack([x["labels"] for x in sorted_batch])

        return {"inputs": sequences_padded, "labels": labels}, sequences_len


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
        for data, data_len in dataloader:
            # デバイスの指定
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            # print(labels)
            # 順伝播
            outputs = model(inputs, data_len)

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
        for data, data_len in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs, data_len)
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

    if opt.label == "label1":
        text_list, label_list = make_dataset(opt.processed_label1_dir, opt.processed_texts_dir)
    else:
        text_list, label_list = make_dataset(opt.processed_label2_dir, opt.processed_texts_dir)

    train_text, val_test_text, train_label, val_test_label = train_test_split(
        text_list, label_list, test_size=0.2, shuffle=True, random_state=123
    )
    val_text, test_text, val_label, test_label = train_test_split(
        val_test_text, val_test_label, test_size=0.5, shuffle=True, random_state=123
    )

    logger.info("Vectorize train text...")
    vectorizer = FeatureVectorizer()
    X_train = vectorizer.make_feature_vector(train_text)
    logger.info("Done.")
    logger.info("Vectorize validation text...")
    X_val = vectorizer.make_feature_vector(val_text)
    logger.info("Done.")
    logger.info("Vectorize test text...")
    X_test = vectorizer.make_feature_vector(test_text)
    logger.info("Done.")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_label)
    y_val = mlb.transform(val_label)
    y_test = mlb.transform(test_label)

    dict = defaultdict(int)
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    for text in X_train:
        for word in text.translate(table).split():
            dict[word] += 1
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    # 単語ID辞書の作成
    word2id = {word: i + 2 for i, (word, cnt) in enumerate(dict) if cnt > 0}

    # IDへ変換
    def tokenizer(text, word2id=word2id, unk=1):
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        return [word2id.get(word, unk) for word in text.translate(table).split()]

    train_text_id = [tokenizer(line) for line in tqdm(X_train)]
    valid_text_id = [tokenizer(line) for line in tqdm(X_val)]
    test_text_id = [tokenizer(line) for line in tqdm(X_test)]

    dataset_train = CreateDataset(train_text_id, y_train)
    dataset_valid = CreateDataset(valid_text_id, y_val)
    dataset_test = CreateDataset(test_text_id, y_test)

    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 2  # 辞書のID数 + パディングID
    EMB_SIZE = 100
    PADDING_IDX = 0
    OUTPUT_SIZE = y_train.shape[1]  # ラベルの総種類数
    OUT_CHANNELS = 200
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 80
    NUM_EPOCHS = 100
    DROP_RATE = 0.1

    print(OUTPUT_SIZE)
    weights = torch.zeros(VOCAB_SIZE, EMB_SIZE)

    model = CNN(
        VOCAB_SIZE,
        EMB_SIZE,
        OUTPUT_SIZE,
        OUT_CHANNELS,
        KERNEL_HEIGHTS,
        STRIDE,
        PADDING,
        DROP_RATE,
        emb_weights=weights,
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
