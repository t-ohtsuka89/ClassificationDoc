import argparse
import glob
import os
import string
from collections import defaultdict

import MeCab
import regex
from sklearn.preprocessing import MultiLabelBinarizer


def load_dataset(labels_dir: str, texts_dir: str):
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


def get_dataset_info(text_dir: str, label_dir: str):
    text_list, label_list = load_dataset(label_dir, text_dir)
    # 分かち書き
    mecab_wakati_tagger = MeCab.Tagger("-Owakati")
    text_list = [mecab_wakati_tagger.parse(text) for text in text_list]

    # MultiLabel Binaraize
    mlb = MultiLabelBinarizer()
    label_list = mlb.fit_transform(label_list)

    word_count: defaultdict[str, int] = defaultdict(int)
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    for text in text_list:
        text: str
        for word in text.translate(table).split():
            word_count[word] += 1
    word_freq_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    word2id: dict[str, int] = {word: i + 2 for i, (word, cnt) in enumerate(word_freq_list) if cnt > 0}

    vocab_size = len(set(word2id.values())) + 2
    output_size = len(mlb.classes_)
    return vocab_size, output_size


def main(args):
    label_dir: str = args.processed_label1_dir if args.label == "label1" else args.processed_label2_dir
    vocab_size, output_size = get_dataset_info(
        text_dir=args.processed_texts_dir,
        label_dir=label_dir,
    )
    print(f"語彙数: {vocab_size}")
    print(f"クラス数: {output_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_texts_dir", default="./data/texts")
    parser.add_argument("--processed_label1_dir", default="./data/label_level1")
    parser.add_argument("--processed_label2_dir", default="./data/label_level2")
    parser.add_argument("--label", choices=["label1", "label2"], default="label1")

    args = parser.parse_args()
    main(args)
