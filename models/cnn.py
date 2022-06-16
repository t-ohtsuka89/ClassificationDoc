import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn


class CNN(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        emb_size: int,
        out_channels: int,
        drop_rate: float,
        learning_rate: float,
        optimizer: str,
        T_max: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_grams = [2, 3, 4]

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        for i in range(len(self.n_grams)):
            conv = nn.Conv2d(
                1,
                out_channels,
                (self.n_grams[i], emb_size),
                padding=(i, 0),
            )
            setattr(self, f"conv_{i}", conv)
        self.drop = nn.Dropout(drop_rate)
        self.output = nn.Linear(out_channels * 3, output_size)

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)

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
