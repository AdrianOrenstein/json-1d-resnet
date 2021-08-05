from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger
import pytorch_lightning as pl
import torch

Json = Dict[str, Any]
json_torch_sample = Tuple[torch.LongTensor, torch.LongTensor]


@dataclass
class JsonSample:
    sample: torch.LongTensor
    label: torch.LongTensor


class PadAllDataToLength:
    """
    Used for

    """

    def __init__(
        self,
        pad_to_length: int,
        pad_val: int,
    ):
        self.pad_to_length = pad_to_length
        self.pad_val = pad_val

    def __call__(self, batch_of_data: List[Tuple[torch.LongTensor, torch.LongTensor]]):
        if len(batch_of_data) <= 1:
            sample, label = batch_of_data

            return (
                torch.cat(
                    (
                        sample,
                        torch.tensor(
                            [self.pad_val] * ((self.pad_to_length) - len(sample))
                        ).long(),
                    )
                )[: self.pad_to_length],
                label,
            )
        else:
            samples: List[torch.LongTensor] = [X.flatten() for X, y in batch_of_data]
            labels: List[torch.LongTensor] = [y for X, y in batch_of_data]

            padded_samples = [
                torch.cat(
                    (
                        sample,
                        torch.tensor(
                            [self.pad_val] * ((self.pad_to_length) - len(sample))
                        ).long(),
                    )
                )[: self.pad_to_length]
                for sample in samples
            ]

            return torch.stack(padded_samples), torch.stack(labels).flatten()


class JSONDataset(torch.utils.data.Dataset):
    """
    Given JSON, return the tokenised string.

    Store dataset as:
    data_directory_path/
      class0/
        file_containing_sample_and_label1.json
        file_containing_sample_and_label2.json

      class1/
        file_containing_sample_and_label1.json
    """

    def __init__(
        self,
        data_directory_path: Path,
        preprocessing: Callable[[Json], json_torch_sample],
        balance_classes: bool = False,
    ):
        """Initialization"""
        self.balance_classes = balance_classes
        self.class_folders = sorted(str(t) for t in data_directory_path.glob("*"))

        self.class_folders = [p for p in self.class_folders]

        self.class_lookup: Dict[str, int] = dict(
            zip(self.class_folders, range(len(self.class_folders)))
        )

        self.dataset_files: List[Path] = self.prep_data(data_directory_path)

        self.preprocessing = preprocessing

    def prep_data(self, path: Path) -> List[Path]:
        """
        Get all the filepaths into a list

        """
        assert path.is_dir()

        all_samples_in_dataset = []
        for class_filepath in self.class_folders:
            class_filepath = class_filepath.split("/")[-1]
            all_samples_in_dataset.extend(list(path.glob(class_filepath + "/*.json")))

        return all_samples_in_dataset

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset_files)

    def get_sample(self, filename: Path) -> JsonSample:
        filename_str = str(filename)
        parent_dir = str(filename.parent)

        with open(filename_str, "r") as f:
            data: Json = json.loads(f.read())

        if not data:
            logger.warning(f"No json extracted from {data} at {filename}")

        return JsonSample(
            sample=self.preprocessing(data),
            label=torch.LongTensor([self.class_lookup[parent_dir]]),
        )

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Generates one sample of data"""

        filename = self.dataset_files[index]
        data = self.get_sample(filename)

        return data.sample, data.label


class JsonDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Path,
        val_data_dir: Path,
        preprocessing: Callable[[Json], List[int]],
        pad_token_id: int,
        batch_size: int,
        sequence_length: int,
        balance_classes: bool,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.preprocessing = preprocessing
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.balance_classes = balance_classes
        self.setup()

    def setup(self):
        self.train_dataset = JSONDataset(
            data_directory_path=self.train_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=self.balance_classes,
        )
        self.val_dataset = JSONDataset(
            data_directory_path=self.val_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=self.balance_classes,
        )

        self.num_classes = len(self.train_dataset.class_folders)
        assert len(self.train_dataset.class_folders) == len(
            self.val_dataset.class_folders
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=PadAllDataToLength(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         collate_fn=PadAllDataToLength(
    #             pad_to_length=self.sequence_length,
    #             pad_val=self.pad_token_id,
    #         ),
    #         num_workers=4,
    #         pin_memory=True,
    #     )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=PadAllDataToLength(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )
