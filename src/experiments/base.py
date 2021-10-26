from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from loguru import logger


class BaseExperiment(pl.LightningModule):
    NAME = "base-experiment"
    TAGS = {}

    def __init__(
        self,
        **kwargs: Optional[Any],
    ):
        super().__init__()

    def calculate_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return NotImplementedError

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Tuple[torch.Tensor, ...]:
        x, y, patient_id = batch

        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)

        if stage != "train":
            loss = loss.mean()

        return x, y, y_hat, loss, patient_id

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        _, y, y_hat, loss, patient_id = self.step(batch, stage="train")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f'{batch_idx} {patient_id} recieved invalid {loss=}')
            return None

        self.log("train_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="train_", prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss, patient_id = self.step(batch, stage="val")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f'{batch_idx} {patient_id} recieved invalid {loss=}')
            return None
        
        self.log("val_loss", loss.item())

        self.log_metrics(y_hat, y, prefix="val_")

    def test_step(self, batch, batch_idx) -> None:
        _, y, y_hat, loss, patient_id = self.step(batch, stage="test")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f'{batch_idx} {patient_id} recieved invalid {loss=}')
            return None

        self.log(
            f"test_loss", loss.item()
        )

        self.log_metrics(
            y_hat,
            y,
            prefix=f"test_",
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

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-06)

        return {
            "optimizer": opt,
        }
