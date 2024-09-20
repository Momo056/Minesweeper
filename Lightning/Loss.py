import torch

from models.Game_Tensor_Interface import Game_Tensor_Interface


class Boundary_KL_Loss:
    def __init__(self, boundary_alpha: float) -> None:
        self.boundary_alpha = boundary_alpha

    def __call__(self, model_output, grid_tensor, mines):
        # Slow compared to training step
        boundary = Game_Tensor_Interface.unknown_boundaries(grid_tensor) # Slow

        part_loss = model_output[:, 0] * (1 - mines * 1) + model_output[:, 1] * (
            mines * 1
        )

        boundary_loss = -torch.mean(part_loss[boundary]) # Slow
        flat_loss = -torch.mean(part_loss[~boundary]) # Slow

        # Kullback leibler divergence
        loss = (
            self.boundary_alpha * boundary_loss + (1 - self.boundary_alpha) * flat_loss
        )
        return loss

