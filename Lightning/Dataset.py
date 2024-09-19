from os import path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class Data_Module(pl.LightningDataModule):
    def __init__(
        self,
        tensor_file_path: str,
        batch_size: int,
        val_size: float = 0.1,
        random_seed=86431,
    ) -> None:
        super().__init__()
        self.tensor_file_path = tensor_file_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        assert path.exists(self.tensor_file_path)

        return super().prepare_data()

    def setup(self, stage: str) -> None:
        dataset = torch.load(self.tensor_file_path)  # Shape : [2990, 10, 8, 8]
        train_data, test_data, train_mines, test_mines = train_test_split(
            *dataset,
            test_size=self.val_size,
            shuffle=False,
            random_state=self.random_seed
        )  # Do not shuffle to not mix the same grids in training and test
        self.train_dataset = TensorDataset(train_data.type(torch.float32), train_mines)
        self.val_dataset = TensorDataset(test_data.type(torch.float32), test_mines)

        return super().setup(stage)

    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
