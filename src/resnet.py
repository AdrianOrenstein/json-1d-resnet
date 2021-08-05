import copy
from pathlib import Path
import string
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
from src.dataset import Json, JsonDatasetDataModule
from src.resnet_key_value import JSONKeyValueResnet_EmbeddingEncoder
import torch


class ClassificationExperiment(pl.LightningModule):
    NAME = "resnet"
    TAGS = {
        "MLFLOW_RUN_NAME": NAME,
        "dataset": "medical-hypertension",
        "algorithm": "resnet",
        "model": "torchvision.models.resnet50_1D",
    }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(ClassificationExperiment.NAME)
        parser.add_argument("--learning_rate", type=float, default=0.003)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--sequence_length", type=int, default=16384)
        parser.add_argument(
            "--train_data_path", type=str
        )
        parser.add_argument(
            "--val_data_path", type=str
        )
        return parent_parser

    def __init__(
        self,
        schema: Json,
        **kwargs: Optional[Any],
    ):
        super().__init__()

        self.learning_rate = kwargs.get("learning_rate")
        self.batch_size = kwargs.get("batch_size")
        self.sequence_length = kwargs.get("sequence_length")
        self.train_data_path = kwargs.get("train_data_path")
        self.val_data_path = kwargs.get("val_data_path")

        self.balanced_dataloaders = kwargs.get("balanced_dataset")

        self.metrics: Dict[str, Callable] = kwargs.get(
            "metrics",
            {
                "Accuracy": pl_metrics.classification.Accuracy(),
                "Precision": pl_metrics.classification.Precision(),
                "Recall": pl_metrics.classification.Recall(),
            },
        )

        self.schema = self.parse_design_summary_schema(schema)

        self.model = JSONKeyValueResnet_EmbeddingEncoder(
            num_classes=2,
            vocab=list(self.schema) + list(string.printable),
            layers=[3, 4, 6, 3],
        )

        self.data_module = JsonDatasetDataModule(
            train_data_dir=Path(self.train_data_path),
            val_data_dir=Path(self.val_data_path),
            preprocessing=copy.deepcopy(self.model.preprocessing),
            pad_token_id=copy.deepcopy(self.model.tokeniser.pad_token_id),
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            balance_classes=self.balanced_dataloaders,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def parse_design_summary_schema(self, schema_json: Dict[str, Any]) -> List[str]:
        return schema_json["schema"]

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-06)

        return {
            "optimizer": opt,
        }

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Tuple[torch.Tensor, ...]:
        x, y, sample_no = batch

        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)

        if stage != "train":
            loss = loss.mean()

        return x, y, y_hat, loss, sample_no

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        _, y, y_hat, loss, sample_no = self.step(batch, stage="train")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"{batch_idx} {sample_no} recieved invalid {loss=}")
            return None

        self.log("train_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="train_", prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss, sample_no = self.step(batch, stage="val")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"{batch_idx} {sample_no} recieved invalid {loss=}")
            return None

        self.log("val_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="val_")

    def test_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss, sample_no = self.step(batch, stage="test")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"{batch_idx} {sample_no} recieved invalid {loss=}")
            return None

        self.log("test_loss", loss.item())

        self.log_metrics(
            y_hat,
            y,
            prefix="test_",
        )

    def log_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        prefix: str,
        prog_bar: bool = False,
    ) -> None:

        for metric_name, metric_function in self.metrics.items():
            if len(y.unique()) != 1:
                self.log(
                    f"{prefix}_{metric_name}",
                    metric_function(torch.argmax(y_hat, dim=1).cpu(), y.cpu()),
                    prog_bar=prog_bar,
                )
