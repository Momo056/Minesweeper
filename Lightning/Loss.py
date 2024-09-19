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

class Boundary_KL_Loss:
    def __init__(self, boundary_alpha: float) -> None:
        self.boundary_alpha = boundary_alpha

    def __call__(self, model_output, grid_tensor, mines):
        boundary = Game_Tensor_Interface.unknown_boundaries(grid_tensor)

        part_loss = model_output[:, 0]*(1-mines*1) + model_output[:, 1]*(mines*1)

        boundary_loss = - torch.mean(part_loss[boundary])
        flat_loss = - torch.mean(part_loss[~boundary])

        # Kullback leibler divergence
        loss = self.boundary_alpha * boundary_loss + (1-self.boundary_alpha)*flat_loss
        return loss
