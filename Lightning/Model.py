import torch
from torchmetrics import Accuracy
from torch import nn, optim
from Lightning.Loss import Boundary_KL_Loss
from models.Symetry_Invariant_Conv2D import Symetry_Inveriant_Conv2D
from models.utils import valid_argmax2D
from models.Game_Tensor_Interface import Game_Tensor_Interface
import pytorch_lightning as pl


class NN(pl.LightningModule):
    def __init__(self, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(10, 16, 9, padding="same", padding_mode="zeros"),
                    nn.ELU(),
                )
            ),
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(16, 16, 9, padding="same", padding_mode="zeros"),
                    nn.ELU(),
                )
            ),
            Symetry_Inveriant_Conv2D(
                nn.Sequential(
                    nn.Conv2d(16, 2, 1, padding="same", padding_mode="zeros"),
                    nn.ELU(),
                )
            ),
            nn.LogSoftmax(-3),
        )
        self.compute_loss = Boundary_KL_Loss(alpha)
        self.game_tensor_interface = Game_Tensor_Interface()
        self.accuracy = Accuracy("binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)

        grid_tensor, mines = batch

        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, model_output = self._common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):  # Get data to cuda if possible
        grid_tensor, mines = batch
        model_output = self.model(grid_tensor)
        loss = self.compute_loss(model_output, grid_tensor, mines)
        return loss, model_output

    def predict_step(self, batch, batch_idx):  # Get data to cuda if possible
        print("Not tested")
        grid_tensor, mines = batch
        grid, grid_view = self.game_tensor_interface.to_grid(grid_tensor)

        model_output = self.model(grid_tensor)

        no_mines_proba = torch.exp(model_output[:, 0])

        # Remove the possibility to pick a non boundary box by giving it negative probability
        boundary = Game_Tensor_Interface.unknown_boundaries(grid_tensor)
        no_mines_proba -= (~boundary) * 1

        # If the remaining boxes are not in the boundary (can append if an area is surronded by mines)
        no_mines_proba += (~boundary) * (
            torch.max(no_mines_proba, dim=0) < self.out_of_boundary_treshold
        )

        return [
            valid_argmax2D(t.to("cpu"), grid_view)
            for t, g in zip(grid_tensor, grid_view)
        ]

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())
