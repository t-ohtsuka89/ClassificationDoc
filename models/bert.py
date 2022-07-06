from typing import cast

import pl_bolts
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_forecasting.optim import Ranger
from special_tokens import SpecialToken
from torch import Tensor, nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from utils.mixout import MixLinear


class Bert(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        model_name: str,
        output_size: int,
        learning_rate: float,
        optimizer: str,
        n_warmup: int,
        num_train_steps: int,
        mixout: float = 0.7,
        padding_idx=SpecialToken.PAD,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bert = cast(
            BertForSequenceClassification | RobertaForSequenceClassification,
            AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=output_size),
        )
        assert isinstance(self.bert, BertForSequenceClassification | RobertaForSequenceClassification)
        if mixout > 0:
            print("Initializing Mixout Regularization")
            for sup_module in self.bert.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                            module.in_features, module.out_features, bias, target_state_dict["weight"], mixout
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)
            print("Done.!")

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.train_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.valid_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(average="micro", threshold=0.5)

    def forward(self, batch_input_ids: Tensor, batch_input_mask: Tensor):
        outputs: SequenceClassifierOutput = self.bert(
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
        weight_decay = 0.01
        layerwise_learning_rate_decay = 0.99
        adam_epsilon = 1e-6

        grouped_optimizer_params = self.get_optimizer_grouped_parameters(weight_decay, layerwise_learning_rate_decay)
        if self.hparams["optimizer"] == "Ranger":
            optimizer = Ranger(
                # grouped_optimizer_params,
                self.parameters(),
                lr=self.hparams["learning_rate"],
                # eps=adam_epsilon,
            )
        else:
            optimizer = torch.optim.__dict__[self.hparams["optimizer"]](
                # grouped_optimizer_params,
                self.parameters(),
                lr=self.hparams["learning_rate"],
                # eps=adam_epsilon,
            )

        if self.hparams["num_train_steps"] is None:
            return optimizer
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams["n_warmup"],
                num_training_steps=self.hparams["num_train_steps"],
            )
            # scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            #     optimizer, warmup_epochs=int(self.hparams["T_max"] * 0.1), max_epochs=self.hparams["T_max"]
            # )
            return [optimizer], [scheduler]

    def create_criterion(self):
        return nn.BCEWithLogitsLoss()

    def get_optimizer_grouped_parameters(self, weight_decay: float, layerwise_learning_rate_decay: float):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert.named_parameters() if "classifier" in n or "pooler" in n],
                "weight_decay": 0.0,
                "lr": self.hparams["learning_rate"],
            },
        ]
        # initialize lrs for every layer
        layers = [getattr(self.bert, self.bert.base_model_prefix).embeddings] + list(
            getattr(self.bert, self.bert.base_model_prefix).encoder.layer
        )
        layers.reverse()
        lr = self.hparams["learning_rate"]
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        return optimizer_grouped_parameters
