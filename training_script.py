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

pl.LightningModule

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

def valid_argmax2D(tensor, grid_view):
    valid_flat = ~torch.flatten(torch.tensor(grid_view))
    output_flat = torch.flatten(tensor)

    valid_indices = torch.where(valid_flat)[0]
    valid_max_index = torch.argmax(output_flat[valid_indices])
    valid_indices[valid_max_index]

    if grid_view.shape[0] != grid_view.shape[1]:
        raise NotImplementedError()
    
    max_row = valid_indices[valid_max_index] // grid_view.shape[0]
    max_col = valid_indices[valid_max_index] % grid_view.shape[0]
    return max_row, max_col

class NN(pl.LightningModule):
    def __init__(self, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(10, 16, 9, padding='same', padding_mode='zeros'),
                    nn.ELU(),
                )
            ),
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros'),
                    nn.ELU(),
                )
            ),
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(16, 2, 1, padding='same', padding_mode='zeros'),
                    nn.ELU(),
                )
            ),
            nn.LogSoftmax(-3),
        )
        self.compute_loss = Boundary_KL_Loss(alpha)
        self.game_tensor_interface = Game_Tensor_Interface()
        self.accuracy = Accuracy('binary')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)

        grid_tensor, mines = batch

        self.log_dict({'train_loss':loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)
        self.log('validation_loss', loss)
        return loss
    
    def _common_step(self, batch, batch_idx):# Get data to cuda if possible
        grid_tensor, mines = batch
        model_output = model(grid_tensor)
        loss = self.compute_loss(model_output, grid_tensor, mines)
        return loss, model_output
    
    def predict_step(self, batch, batch_idx):# Get data to cuda if possible
        print('Not tested')
        grid_tensor, mines = batch
        grid, grid_view = self.game_tensor_interface.to_grid(grid_tensor)
        
        model_output = self.model(grid_tensor)

        no_mines_proba = torch.exp(model_output[:, 0])

        # Remove the possibility to pick a non boundary box by giving it negative probability
        boundary = Game_Tensor_Interface.unknown_boundaries(grid_tensor)
        no_mines_proba -= (~boundary)*1

        # If the remaining boxes are not in the boundary (can append if an area is surronded by mines)
        no_mines_proba += (~boundary)*(torch.max(no_mines_proba, dim=0) < self.out_of_boundary_treshold)
        
        return [valid_argmax2D(t.to('cpu'), grid_view) for t, g in zip(grid_tensor, grid_view)]
    
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
alpha = 0.95 # Part of loss on the Unknown boundaries


# Initialize network
model = NN(alpha).to(device)
dm = Data_Module(
    'dataset/lose_bot/12x12_23253.pt',
    batch_size,
    0.1,
)
trainer = pl.Trainer(
    accelerator='gpu', # 'gpu' or 'tpu'
    devices=[0], # Devices to use
    min_epochs=1, 
    max_epochs=5, 
    precision=16,
    # overfit_batches=1, # Debug : Try to overfit the model to one batch
    fast_dev_run=True, # Debug : Smaller loops
)
trainer.fit(model, dm)
trainer.validate(model, dm)


