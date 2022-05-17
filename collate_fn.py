import torch
from torch.nn.utils.rnn import pad_sequence


class Padsequence:
    """Dataloaderからミニバッチを取り出すごとに最大系列長でパディング"""

    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        inputs = [sample["input"] for sample in batch]
        labels_list = [sample["labels"] for sample in batch]
        padded_inputs = pad_sequence(inputs, batch_first=True)  # padding
        return {
            "input": padded_inputs.contiguous(),
            "labels": torch.stack(labels_list).contiguous(),
            "lengths": torch.tensor(list(map(len, inputs)), dtype=torch.int64),
        }
