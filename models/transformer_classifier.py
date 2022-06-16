import pl_bolts
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn

from .transformer import TransformerModel


class TransformerClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        emb_size: int,
        nhead: int,
        nlayers: int,
        hidden_dim: int,
        drop_rate: float,
        optimizer: str,
        learning_rate: float,
        mode: str,
        T_max: int | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TransformerModel(
            ntoken=vocab_size,
            d_model=emb_size,
            nhead=nhead,
            d_hid=hidden_dim,
            nlayers=nlayers,
            dropout=drop_rate,
            padding_idx=padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.Mish(),
            # nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, output_size),
        )

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)

    def forward(self, x: Tensor, seq_len: Tensor):
        x = self.encoder(x)
        if self.hparams["mode"] == "max":
            x = torch.max(x, dim=1)[0]
        elif self.hparams["mode"] == "mean":
            # x = torch.mean(x, dim=1)
            x = torch.sum(x, dim=1)
            x = torch.div(x.transpose(0, 1), seq_len.to(x.device)).transpose(0, 1)
        elif self.hparams["mode"] == "special_token":
            x = x[:, 0]
        else:
            raise ValueError(f"Unexpected Mode: {self.hparams['mode']}")
        out = self.classifier(x)
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
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=10, max_epochs=self.hparams["T_max"]
            )
            return [optimizer], [scheduler]

    def create_criterion(self):
        return nn.BCEWithLogitsLoss()
