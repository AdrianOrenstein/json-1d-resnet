from pathlib import Path
from typing import Dict, List, Optional, Set, Type

import pytorch_lightning as pl
from src.experiments import (
    json_1d_resnet,
)
from src.experiments.base import BaseExperiment

AnExperiment = Type[BaseExperiment]
_EXPERIMENT_CLASSES: List[AnExperiment] = [
    json_1d_resnet.ClassificationExperiment,
]
_EXPERIMENTS: Dict[str, AnExperiment] = {c.NAME: c for c in _EXPERIMENT_CLASSES}


def get_experiment_names() -> Set[str]:
    return set(_EXPERIMENTS.keys())


def get_experiment(
    experiment_name: str, checkpoint_path: Optional[Path] = None
) -> pl.Trainer:

    experiment: Optional[pl.Trainer] = _EXPERIMENTS.get(experiment_name, None)

    if not experiment:
        raise KeyError(f"{experiment_name} unknown, available {get_experiment_names()}")

    if checkpoint_path is None:
        return experiment

    else:
        return experiment.load_from_checkpoint(str(checkpoint_path))
