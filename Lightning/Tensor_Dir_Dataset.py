from enum import Enum
from math import ceil
from os import listdir, path
import re
from typing import Iterable
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data import random_split, Dataset, ConcatDataset
import pytorch_lightning as pl

class AGREGATION_STRATEGY(Enum):
    UNIFORM_DATASET_WEIGHT = 1


class Tensor_Dir_Dataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        val_size: float = 0.1,
        test_size: float = 0.1,
        dataset_root: str = 'dataset',
        agregation_strategy: AGREGATION_STRATEGY = AGREGATION_STRATEGY.UNIFORM_DATASET_WEIGHT,
        random_seed=86431,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.generator = torch.Generator().manual_seed(random_seed)
        self.dataset_root = dataset_root
        self.agregation_strategy = agregation_strategy

    def prepare_data(self) -> None:
        dir_paths = self.get_data_folders()

        all_datasets = self.load_all_datasets(dir_paths)

        # Assume the same grid size
        # Need to implement the case of multiple grid_size

        self.merge_datasets(all_datasets)

        return super().prepare_data()
    
    def load_all_datasets(self, dir_path_list: list[str]):
        return [
            self.load_dataset(d)
            for d in dir_path_list
        ]

    def load_dataset(self, dir_path: str):
        grid_mines = [
            torch.load(path.join(dir_path, t_name)) 
            for t_name in listdir(dir_path) 
            if t_name.endswith('.pt')
        ]

        grid_tensors = torch.cat([g for g, m in grid_mines], dim=0).type(torch.float32)
        mines_tensors = torch.cat([m for g, m in grid_mines], dim=0)

        return TensorDataset(grid_tensors, mines_tensors)

    def get_data_folders(self):
        return [
            path.join(self.dataset_root, d) for d in listdir(self.dataset_root)
            if re.match(r'\d*x\d*_m\d*', d)
        ]

    def merge_datasets(self, dataset_list: list[Dataset]):
        if self.agregation_strategy != AGREGATION_STRATEGY.UNIFORM_DATASET_WEIGHT:
            raise NotImplementedError(self.agregation_strategy)
        
        # Use the same number of entry for every classes of val and test to have a more balanced measurment
        val_size = ceil(min([len(d) for d in dataset_list]) * self.val_size)
        test_size = ceil(min([len(d) for d in dataset_list]) * self.test_size)

        # Dataset splits
        train_dataset_list = []
        val_dataset_list = []
        test_dataset_list = []
        for d in dataset_list:
            splits = random_split(d, [len(d)-val_size-test_size, val_size, test_size], self.generator)

            train_dataset_list.append(splits[0])
            val_dataset_list.append(splits[1])
            test_dataset_list.append(splits[2])

        self.train_dataset = ConcatDataset(train_dataset_list)
        self.val_dataset = ConcatDataset(val_dataset_list)
        self.val_dataset = ConcatDataset(test_dataset_list)

        # Train sampling
        train_sizes = [len(d) for d in train_dataset_list]
        train_weights = torch.cat([torch.ones(ts)/ts for ts in train_sizes])
        max_sample = len(dataset_list) * min(train_sizes) # Since we do without replacement, we need to sample the smallest dataset accordingly
        self.train_sampler = WeightedRandomSampler(train_weights, max_sample, replacement=False, generator=self.generator)


    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)