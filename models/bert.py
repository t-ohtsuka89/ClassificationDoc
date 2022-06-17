from typing import cast

import pl_bolts
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_forecasting.optim import Ranger
from torch import Tensor, nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertForSequenceClassification


class Bert(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        model_name: str,
        output_size: int,
        learning_rate: float,
        optimizer: str,
        T_max: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = cast(
            BertForSequenceClassification,
            BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        # BERTの最後の層だけ更新ON
        for param in self.model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        # クラス分類のところもON
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)

    def forward(self, batch_input_ids: Tensor, batch_input_mask: Tensor):
        outputs: SequenceClassifierOutput = self.model(
            batch_input_ids,
            token_type_ids=None,
            attention_mask=batch_input_mask,
        )

        return outputs.logits

    def training_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        mask = batch["mask"]
        labels = batch["labels"]
        logits = self.forward(inputs, mask)
        loss: Tensor = self.criterion(
            logits.view(-1, self.hparams["output_size"]), labels.type_as(logits).view(-1, self.hparams["output_size"])
        )
        preds = torch.sigmoid(logits)
        self.train_f1(preds, labels.to(dtype=torch.int))
        self.log("train_loss", loss)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        mask = batch["mask"]
        labels = batch["labels"]
        logits = self.forward(inputs, mask)
        preds = torch.sigmoid(logits)
        loss: Tensor = self.criterion(
            logits.view(-1, self.hparams["output_size"]), labels.type_as(logits).view(-1, self.hparams["output_size"])
        )
        self.valid_f1(preds, labels.to(dtype=torch.int))
        self.log("val_loss", loss)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        mask = batch["mask"]
        labels = batch["labels"]
        logits = self.forward(inputs, mask)
        preds = torch.sigmoid(logits)
        self.test_f1(preds, labels.to(dtype=torch.int))
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "Ranger":
            optimizer = Ranger(
                [
                    {"params": self.model.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                    {"params": self.model.classifier.parameters(), "lr": self.hparams["learning_rate"]},
                ]
            )
        else:
            optimizer = torch.optim.__dict__[self.hparams["optimizer"]](
                self.parameters(),
                lr=self.hparams["learning_rate"],
            )

        if self.hparams["T_max"] is None:
            return optimizer
        else:
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=int(self.hparams["T_max"] * 0.1), max_epochs=self.hparams["T_max"]
            )
            return [optimizer], [scheduler]

    def create_criterion(self):
        return nn.BCEWithLogitsLoss()
