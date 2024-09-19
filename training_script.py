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
from Lightning.Model import NN
from Lightning.Dataset import Data_Module
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

import Lightning.config as cfg


# Initialize network
model = NN(cfg.ALPHA)
dm = Data_Module(
    cfg.TENSOR_DATA_FILE,
    cfg.BATCH_SIZE,
    cfg.VAL_SIZE,
)
trainer = pl.Trainer(
    accelerator=cfg.ACCELERATOR, # 'gpu' or 'tpu'
    devices=cfg.TRAINER_DEVICES, # Devices to use
    min_epochs=cfg.MIN_EPOCHS, 
    max_epochs=cfg.MAX_EPOCHS, 
    precision=cfg.PRECISION,
    # overfit_batches=1, # Debug : Try to overfit the model to one batch
    fast_dev_run=True, # Debug : Smaller loops
)
trainer.fit(model, dm)
trainer.validate(model, dm)


