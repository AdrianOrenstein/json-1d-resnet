from argparse import ArgumentParser
import time

import pytorch_lightning as pl
from src.dataset.json_files import JsonDatasetDataModule
from src.experiments.experiments import get_experiment

def get_experiment_from_args(parser):
    args, unknown = parser.parse_known_args()
    return get_experiment(args.experiment_name)

def train():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="")
    experiment = get_experiment_from_args(parser)
    parser = experiment.add_model_specific_args(parser)
    parser = JsonDatasetDataModule.add_data_module_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--SEED",
        type=int,
        default=int(time.time()),
    )
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    experiment = experiment(
        **vars(args),
    )

    trainer.fit(experiment, experiment.data_module)
    trainer.save_checkpoint(f"weights/{int(time.time())}.ckpt")


if __name__ == "__main__":
    train()
