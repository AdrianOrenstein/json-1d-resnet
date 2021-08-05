from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
import pytorch_lightning as pl
import torch
import glob

Json = Dict[str, Any]
json_torch_sample = Tuple[torch.LongTensor, torch.LongTensor]

from src.dataset.utils import dfs_unpack_json
import concurrent.futures

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
        self.data_directory_path = Path(data_directory_path)
        self.class_folders = sorted(str(t) for t in self.data_directory_path.glob("*"))

        self.class_folders = [p for p in self.class_folders]

        self.class_lookup: Dict[str, int] = dict(
            zip(self.class_folders, range(len(self.class_folders)))
        )

        self.dataset_files: List[Path] = self.prep_data(self.data_directory_path)

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
    @staticmethod
    def add_data_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('JsonDatasetDataModule')
        parser.add_argument("--train_data_path", type=str)
        parser.add_argument("--test_data_path", type=str)
        return parent_parser

    @staticmethod
    def unpack_flattened_keys(file_path: Path):

        with open(file_path, 'r') as f:
            sample = json.load(f)
        
        return set(key for key, _ in dfs_unpack_json(sample['data']))
    
    @staticmethod
    def generate_flattened_json_schema(list_of_sample_filepaths: List[Path], schema_path: str, write_to_path: bool = True, **kwargs) -> Json:
        all_keys = set()
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(JsonDatasetDataModule.unpack_flattened_keys, filepath)
                for filepath in list_of_sample_filepaths
            ]

            for future in concurrent.futures.as_completed(futures):
                for key in future.result():   
                    all_keys.add(key)
        schema = {
            'schema': list(all_keys)
        }

        if write_to_path:
            with open(schema_path, 'w') as f:
                json.dump(schema, f)

        return schema

    def __init__(
        self,
        preprocessing: Callable[[Json], List[int]],
        pad_token_id: int,
        batch_size: int,
        sequence_length: int,
        balance_classes: bool = False,
        **kwargs: Optional[Any],
    ):
        super().__init__()
        self.preprocessing = preprocessing
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.balance_classes = balance_classes

        self.sequence_length: int = kwargs.get("sequence_length")
        self.train_data_dir: Path = kwargs.get("train_data_path")
        self.test_data_dir: Path = kwargs.get("test_data_path")

        logger.info(f'{self.train_data_dir=}\n{self.test_data_dir=}')
        self.setup()

    def setup(self):
        self.train_dataset = JSONDataset(
            data_directory_path=self.train_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=self.balance_classes,
        )
        self.test_dataset = JSONDataset(
            data_directory_path=self.test_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=self.balance_classes,
        )

        self.num_classes = len(self.train_dataset.class_folders)
        assert len(self.train_dataset.class_folders) == len(
            self.test_dataset.class_folders
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

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=PadAllDataToLength(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            pin_memory=True,
        )


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--schema_path", type=str, required=True)
    parser.add_argument("--write_to_path", type=bool, required=True)
    
    parser = JsonDatasetDataModule.add_data_module_specific_args(parser)

    args = parser.parse_args()

    data_module = JsonDatasetDataModule(
        # these kwargs aren't used
        preprocessing = lambda: list,
        pad_token_id = 0,
        batch_size = 16,
        sequence_length = 2**10,
        **vars(args),
    )

    
    all_dataset_files: List[Path] = data_module.train_dataset.dataset_files + data_module.test_dataset.dataset_files

    JsonDatasetDataModule.generate_flattened_json_schema(
        list_of_sample_filepaths = all_dataset_files, **vars(args)
    )