import io
from sys import prefix
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from torchmetrics import Accuracy
from torch import Tensor, nn, optim
import torchvision.transforms.functional
from Lightning.Accumulator_Accuracy import Accumulator_Accuracy
from Lightning.Loss import Boundary_KL_Loss
from models.Symetry_Invariant_Conv2D import Symetry_Inveriant_Conv2D
from models.utils import valid_argmax2D
from models.Game_Tensor_Interface import Game_Tensor_Interface
import pytorch_lightning as pl
import torchvision
from PIL import Image


class NN(pl.LightningModule):
    def __init__(self, alpha: float, out_of_boundary_treshold: float = 0.5, 
                 n_sym_block: int = 3, layer_per_block: int = 2, latent_dim: int = 64, 
                 kernel_size: int = 3, batch_norm_period: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss = Boundary_KL_Loss(alpha)
        self.game_tensor_interface = Game_Tensor_Interface()
        self.accuracy = Accuracy("binary")
        self.out_of_boundary_treshold = out_of_boundary_treshold
        self.custom_accuracy = Accumulator_Accuracy()

        # Call the build_model method with the provided parameters
        self.build_model(n_sym_block, layer_per_block, latent_dim, kernel_size, batch_norm_period)

    def build_model(self, n_sym_block: int, layer_per_block: int, latent_dim: int, kernel_size: int, batch_norm_period: int):
        sym_blocks = []
        for i_block in range(n_sym_block):
            # Symmetric block construction
            sub_blocks = []
            for l in range(layer_per_block):
                first_layer = i_block == 0 and l == 0
                last_layer = i_block != n_sym_block - 1 and l != layer_per_block - 1

                # Sub block construction
                sub_blocks.append(nn.Conv2d(
                    10 if first_layer else latent_dim, 
                    2 if last_layer else latent_dim, 
                    kernel_size, 
                    padding="same", 
                    padding_mode="zeros"
                ))

                if batch_norm_period > 0 and l % batch_norm_period == 0 and not last_layer:
                    sub_blocks.append(nn.BatchNorm2d(latent_dim))
                
                if not last_layer:  # Add activation only for intermediate layers
                    sub_blocks.append(nn.ReLU())
            
            sym_blocks.append(Symetry_Inveriant_Conv2D(nn.Sequential(*sub_blocks)))
        
        self.model = nn.Sequential(*sym_blocks, nn.LogSoftmax(-3))


    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):  # Get data to cuda if possible
        grid_tensor, mines = batch
        model_output = self.model(grid_tensor)
        loss = self.compute_loss(model_output, grid_tensor, mines)
        return loss, model_output
    
    def training_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)

        self.log_dict({"train/loss": loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Output computation
        grid_tensor, mines = batch
        loss, model_output = self._common_step(batch, batch_idx)

        # Prediction accuracy
        success, total = self.compute_success(self.compute_actions(grid_tensor, model_output), mines)
        self.custom_accuracy.update(success, total)

        # Log
        self.log_dict({'val/'+k:v for k, v in {
            'loss': loss,
            'accuracy': self.custom_accuracy,
        }.items()}, on_step=False, on_epoch=True)

        if batch_idx == 0:
            # Log grid
            try:
                self.log_grid(model_output, batch, batch_idx)
            except ValueError as e:
                self.log('plt_error', str(e))
            
        return loss


    def test_step(self, batch, batch_idx):
        # Output computation
        grid_tensor, mines = batch
        loss, model_output = self._common_step(batch, batch_idx)

        # Prediction accuracy
        success, total = self.compute_success(self.compute_actions(grid_tensor, model_output), mines)
        self.custom_accuracy.update(success, total)

        # Log
        self.log_dict({'test/'+k:v for k, v in {
            'loss': loss,
            'accuracy': self.custom_accuracy,
        }.items()}, on_step=False, on_epoch=True)

        return loss
    
    def log_grid(self, model_output, batch, batch_idx):
        grid_tensor, mines = batch
        plot_idx = batch_idx % len(batch)

        # Grid logging
        state_img = Game_Tensor_Interface.view_grid_tensor(grid_tensor[plot_idx].detach().to('cpu'), mines[plot_idx].detach().to('cpu'), view_grid_kwargs={'close_plot':True})

        # Activation
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig: Figure
        ax: Axes

        proba_plot = ax.imshow(torch.exp(model_output[plot_idx, 1].detach().to('cpu')), vmin=0, vmax=1)
        fig.colorbar(proba_plot, ax=ax, label='Mine probability')

        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        activation_img = Image.open(buf)

        plt.close(fig)

        # Paste
        log_img = self.collage_images(state_img, activation_img)

        # Log
        self.logger.experiment.add_image('state_activation_sample', torchvision.transforms.functional.to_tensor(log_img), self.global_step)

        return
    
    def collage_images(self, image1, image2):
        # Get the dimensions of the images
        width1, height1 = image1.size
        width2, height2 = image2.size

        # The height of the collage will be the maximum height of the two images
        collage_height = max(height1, height2)
        # The width of the collage will be the sum of the widths of the two images
        collage_width = width1 + width2

        # Create a new white image (or black background, if needed) for the collage
        collage = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

        # Paste the first image on the left
        collage.paste(image1, (0, 0))

        # Paste the second image on the right
        collage.paste(image2, (width1, 0))

        return collage

    
    def compute_success(self, actions: Tensor, mines: Tensor):
        # Compute the total number of errors by checking the mine at the predicted max indices
        # Use advanced indexing instead of loops
        predicted_mine_status = mines[torch.arange(mines.size(0)), actions[:, 0], actions[:, 1]]

        # The error is when we predict a cell without a mine, but there's actually a mine
        errors = predicted_mine_status.int().sum()

        success = len(mines) - errors

        return success, len(mines)
    
    def compute_actions(self, grid_tensor: Tensor, model_output: Tensor):
        no_mines_proba = torch.exp(model_output[:, 0])

        # Mask out non-boundary boxes by setting their probability to -1
        boundary_mask = Game_Tensor_Interface.unknown_boundaries(grid_tensor)
        no_mines_proba = no_mines_proba.masked_fill(~boundary_mask, -1)

        # Mask out discovered boxes by setting their probability to -2
        discovered_mask = grid_tensor[:, 9] == 0  # discovered cells have value 1, so this is inverted
        no_mines_proba = no_mines_proba.masked_fill(discovered_mask, -2)

        # Get 2D indices of the maximum probabilities for each grid in the batch
        max_vals, max_indices_row = torch.max(no_mines_proba, dim=-1)  # max along width
        max_vals, max_indices_col = torch.max(max_vals, dim=-1)        # max along height

        # Combine row and column indices into 2D coordinates
        max_indices = torch.stack([max_indices_col, 
                                torch.gather(max_indices_row, 1, max_indices_col.unsqueeze(1)).squeeze(-1)], dim=-1)
        
        return max_indices



    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())
