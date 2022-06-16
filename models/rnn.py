import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn


class RNN(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        drop_rate: float,
        learning_rate: float,
        optimizer: str,
        T_max: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.drop1 = nn.Dropout(drop_rate)
        self.bilstm = torch.nn.LSTM(emb_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.drop2 = nn.Dropout(drop_rate)
        self.output = nn.Linear(hidden_size * 2, output_size)

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)

    def forward(self, x: Tensor, seq_len: Tensor):
        embedded_padded_sequence: Tensor = self.emb(x)
        embedded_padded_sequence = self.drop1(embedded_padded_sequence)
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_padded_sequence, seq_len, batch_first=True, enforce_sorted=False
        )
        packed_lstm_out, (hidden, cell) = self.bilstm(packed_x)
        lstm_out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        out: Tensor = lstm_out[:, -1]
        out = self.drop2(out)
        out = self.output(out)
        return out

    def training_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        lengths = batch["lengths"].cpu()
        outputs = self.forward(inputs, lengths)
        preds = torch.sigmoid(outputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        self.train_f1(preds, labels.to(dtype=torch.int))
        self.log("train_loss", loss)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        lengths = batch["lengths"].cpu()
        outputs = self.forward(inputs, lengths)
        preds = torch.sigmoid(outputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        self.valid_f1(preds, labels.to(dtype=torch.int))
        self.log("val_loss", loss)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        lengths = batch["lengths"].cpu()
        outputs = self.forward(inputs, lengths)
        preds = torch.sigmoid(outputs)
        self.test_f1(preds, labels.to(dtype=torch.int))
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.__dict__[self.hparams["optimizer"]](
            self.parameters(),
            lr=self.hparams["learning_rate"],
        )

        if self.hparams["T_max"] is None:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["T_max"],
                eta_min=1e-5,
                last_epoch=-1,
            )
            return [optimizer], [scheduler]

    def create_criterion(self):
        return nn.BCEWithLogitsLoss()
