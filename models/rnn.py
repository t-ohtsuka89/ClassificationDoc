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

        self.encoder = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.bilstm = torch.nn.LSTM(
            emb_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=drop_rate,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size * 2, output_size),
        )

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        lstm_out, (hidden, cell) = self.bilstm(x)
        bilstm_out = torch.cat([hidden[0], hidden[1]], dim=1)
        out: Tensor = self.classifier(bilstm_out)
        return out

    def training_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        outputs = self.forward(inputs)
        preds = torch.sigmoid(outputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        self.train_f1(preds, labels.to(dtype=torch.int))
        self.log("train_loss", loss)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        outputs = self.forward(inputs)
        preds = torch.sigmoid(outputs)
        loss: torch.Tensor = self.criterion(outputs, labels)
        self.valid_f1(preds, labels.to(dtype=torch.int))
        self.log("val_loss", loss)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        labels = batch["labels"]
        outputs = self.forward(inputs)
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
