from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
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

# class NN(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = nn.Sequential(
#             Symetry_Inveriant_Conv2D(
#                 nn.Sequential(
#                     nn.Conv2d(10, 16, 9, padding='same', padding_mode='zeros'),
#                     nn.ELU(),
#                 )
#             ),
#             Symetry_Inveriant_Conv2D(
#                 nn.Sequential(
#                     nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros'),
#                     nn.ELU(),
#                 )
#             ),
#             Symetry_Inveriant_Conv2D(
#                 nn.Sequential(
#                     nn.Conv2d(16, 2, 1, padding='same', padding_mode='zeros'),
#                     nn.ELU(),
#                 )
#             ),
#             nn.LogSoftmax(-3),
#         )
    
#     def forward(self, x):
#         return self.model(x)

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
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)
    
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


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
n_epoch = 3
batch_size = 128
alpha = 0.95 # Part of loss on the Unknown boundaries

# Load Data
dataset = torch.load('dataset/uniform_bot/8x8_2990.pt') # Shape : [2990, 10, 8, 8]
random_seed = 86431
train_data, test_data, train_mines, test_mines = train_test_split(*dataset, test_size=0.1, shuffle=False, random_state=random_seed) # Do not shuffle to not mix the same grids in training and test
train_dataset = TensorDataset(train_data.type(torch.float32).to(device), train_mines.to(device))
test_dataset = TensorDataset(test_data.type(torch.float32).to(device), test_mines.to(device))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = NN(alpha).to(device)

# Loss and optimizer
criterion = Boundary_KL_Loss(alpha)
optimizer = optim.Adam(model.parameters())


# Training loop
for e in tqdm(range(n_epoch)):
    model.train()
    for batch_idx, (grid_tensor, mines) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        grid_tensor = grid_tensor.to(device)
        mines = mines.to(device)

        # Forward
        model_output = model(grid_tensor)
        loss = criterion(model_output, grid_tensor, mines)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()



# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    running_loss = 0.0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for grid_tensor, mines in loader:

            # Move data to device
            grid_tensor = grid_tensor.to(device)
            mines = mines.to(device)

            # Forward pass
            model_output = model(grid_tensor)
            loss = criterion(model_output, grid_tensor, mines)

            running_loss += float(loss.detach().to('cpu'))

    model.train()
    return running_loss / len(loader)


# Check accuracy on training & test to see how good our model
model.to(device)
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")



