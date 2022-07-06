from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_forecasting.optim import Ranger
from torch import Tensor, nn
from utils.mixout import MixLinear

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.big_bird.modeling_big_bird import BigBirdForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.optimization import get_cosine_schedule_with_warmup


class Transformers(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        model_name: str,
        output_size: int,
        learning_rate: float,
        optimizer: str,
        mixout: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # model
        self.set_model()

        # criterion
        self.criterion = self.create_criterion()

        # metrics
        self.set_metrics()

    def set_model(self):
        # load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams["model_name"],
            num_labels=self.hparams["output_size"],
        )
        assert isinstance(
            self.model,
            BertForSequenceClassification | RobertaForSequenceClassification | BigBirdForSequenceClassification,
        )

        # mixout
        mixout_p: float = self.hparams["mixout"]
        if mixout_p > 0:
            self.enable_mixout(mixout_p)

    def set_metrics(self):
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

        # calculator metrics
        self.train_f1(preds, labels.to(dtype=torch.int))

        # logging
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

        # calculator metrics
        self.valid_f1(preds, labels.to(dtype=torch.int))

        # logging
        self.log("val_loss", loss)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx):
        inputs = batch["input"]
        mask = batch["mask"]
        labels = batch["labels"]
        logits = self.forward(inputs, mask)
        preds = torch.sigmoid(logits)

        # calculator metrics
        self.test_f1(preds, labels.to(dtype=torch.int))

        # logging
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        weight_decay = 0.01
        layerwise_learning_rate_decay = 0.99
        adam_epsilon = 1e-6

        if self.hparams.get("enable_lldr", False):
            optimizer_config = {
                "params": self.get_optimizer_grouped_parameters(weight_decay, layerwise_learning_rate_decay),
                "lr": self.hparams["learning_rate"],
                "eps": adam_epsilon,
            }
        else:
            optimizer_config = {
                "params": self.parameters(),
                "lr": self.hparams["learning_rate"],
                "weight_decay": 1e-3,
            }
        if self.hparams["optimizer"] == "Ranger":
            optimizer = Ranger(**optimizer_config)
        else:
            optimizer: torch.optim.Optimizer = torch.optim.__dict__[self.hparams["optimizer"]](**optimizer_config)
        n_warmup: Optional[int] = self.hparams.get("n_warmup", None)
        num_train_steps: Optional[int] = self.hparams.get("num_train_steps", None)
        enable_scheduler = n_warmup is not None
        if enable_scheduler:
            if num_train_steps is None:
                raise ValueError("need num_train_steps is not None")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=n_warmup,
                num_training_steps=num_train_steps,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def create_criterion(self):
        return nn.BCEWithLogitsLoss()

    def get_optimizer_grouped_parameters(self, weight_decay: float, layerwise_learning_rate_decay: float):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "classifier" in n or "pooler" in n],
                "weight_decay": 0.0,
                "lr": self.hparams["learning_rate"],
            },
        ]
        # initialize lrs for every layer
        layers = [getattr(self.model, self.model.base_model_prefix).embeddings] + list(
            getattr(self.model, self.model.base_model_prefix).encoder.layer
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

    def enable_mixout(self, mixout_p: float):
        self.model: BertForSequenceClassification | RobertaForSequenceClassification
        for sup_module in self.model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features,
                        module.out_features,
                        bias,
                        target_state_dict["weight"],
                        mixout_p,
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
