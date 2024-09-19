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

# Set device cuda for GPU if it's available otherwise run on the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINER_DEVICES = [0]

# Hyperparameters
BATCH_SIZE = 128
ALPHA = 0.95 # Part of loss on the Unknown boundaries
VAL_SIZE = 0.1
TENSOR_DATA_FILE = 'dataset/lose_bot/12x12_23253.pt'
ACCELERATOR = 'gpu'
PRECISION=16
MIN_EPOCHS=1 
MAX_EPOCHS=5 

