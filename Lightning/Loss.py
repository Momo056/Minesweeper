import torch
from torch import Tensor
import torch.nn.functional as F
from models.Game_Tensor_Interface import Game_Tensor_Interface


class Boundary_KL_Loss:
    def __init__(self, boundary_alpha: float) -> None:
        self.boundary_alpha = boundary_alpha

    def get_boundary(self, grid_tensor: Tensor):
        return Game_Tensor_Interface.unknown_boundaries(grid_tensor)

    def __call__(self, model_output, grid_tensor, mines):
        # Cross entropy
        part_loss = model_output[:, 0] * (1 - mines * 1) + model_output[:, 1] * (mines * 1)

        boundary = self.get_boundary(grid_tensor)

        return self.compute_boundary_loss(part_loss, boundary)
    
    def compute_boundary_loss(self, part_loss: Tensor, boundary: Tensor):
        boundary_loss = -torch.mean(part_loss[boundary]) # Slow
        flat_loss = -torch.mean(part_loss[~boundary]) # Slow

        # Kullback leibler divergence
        loss = (
            self.boundary_alpha * boundary_loss + (1 - self.boundary_alpha) * flat_loss
        )
        return loss
    
class Fast_Boundary_KL_Loss(Boundary_KL_Loss):
    def compute_boundary_loss(self, part_loss: Tensor, boundary: Tensor):
        return torch.mean(
            part_loss * (1-self.boundary_alpha) 
            + boundary*1.0 * part_loss * (2*self.boundary_alpha - 1)
        )

