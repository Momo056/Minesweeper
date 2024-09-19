from os import path
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.utils.data import random_split
from importlib import reload
from models.Symetry_Invariant_Conv2D import Symetry_Inveriant_Conv2D
from src.Game import Game
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.UI.GUI_Bot_Inputs import GUI_Bot_Inputs
from src.UI.No_UI import No_UI
from src.Grid import Grid
from models.Game_Tensor_Interface import Game_Tensor_Interface
from src.UI.GUI_User_Inputs import GUI_User_Inputs
from src.UI.Command_Line_UI import Command_Line_UI
import src.Players.Minesweeper_bot as mb
import pytorch_lightning as pl


class Data_Module(pl.LightningDataModule):
    def __init__(self, tensor_file_path: str, batch_size: int, val_size: float = 0.1, random_seed = 86431) -> None:
        super().__init__()
        self.tensor_file_path = tensor_file_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        assert path.exists(self.tensor_file_path)

        return super().prepare_data()
    
    def setup(self, stage: str) -> None:
        dataset = torch.load(self.tensor_file_path) # Shape : [2990, 10, 8, 8]
        train_data, test_data, train_mines, test_mines = train_test_split(*dataset, test_size=self.val_size, shuffle=False, random_state=self.random_seed) # Do not shuffle to not mix the same grids in training and test
        self.train_dataset = TensorDataset(train_data.type(torch.float32), train_mines)
        self.val_dataset = TensorDataset(test_data.type(torch.float32), test_mines)

        return super().setup(stage)
    
    def train_dataloader(self) -> torch.Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
