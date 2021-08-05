from argparse import ArgumentParser
import json
import time

import pytorch_lightning as pl
from src.dataset import JsonDatasetDataModule
from src.resnet import ClassificationExperiment


def train():
    parser = ArgumentParser()
    experiment = ClassificationExperiment
    parser = experiment.add_model_specific_args(parser)
    parser = JsonDatasetDataModule.add_data_module_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--SEED",
        type=int,
        default=int(time.time()),
    )
    args = parser.parse_args()

    schema = json.loads("schema.json")

    trainer = pl.Trainer.from_argparse_args(args)
    experiment = experiment(
        **vars(args),
        schema=schema,
    )

    trainer.fit(experiment, experiment.data_module)
    trainer.save_checkpoint(f"weights/{int(time.time())}.ckpt")


if __name__ == "__main__":
    train()
