from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Module
from src.Players.Player_Interface import Player_Interface
from models.Game_Tensor_Interface import Game_Tensor_Interface

class Model_Based_Player(Player_Interface):
    def __init__(self, model: Module, model_deivce='cuda', out_of_boundary_treshold:float=0.1) -> None:
        super().__init__()
        self.tensor_interface = Game_Tensor_Interface()
        self.model = model
        self.model_deivce = model_deivce
        self.out_of_boundary_treshold = out_of_boundary_treshold
        self.last_map: Tensor = None

    def valid_argmax2D(self, tensor: Tensor, grid_view: ndarray):
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

    def action(self, grid: ndarray, grid_view: ndarray) -> tuple[int, int]:
        tensor_representation = self.tensor_interface.to_tensor(grid, grid_view).type(torch.float32).to(self.model_deivce)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(tensor_representation.reshape((1, *tensor_representation.shape)))[0]

        no_mines_proba = torch.exp(model_output[0])
        self.last_map = no_mines_proba

        # Remove the possibility to pick a non boundary box by giving it negative probability
        boundary = Game_Tensor_Interface.unknown_boundaries(tensor_representation.reshape(1, *tensor_representation.shape))[0]
        no_mines_proba -= (~boundary)*1

        # If the remaining boxes are not in the boundary (can append if an area is surronded by mines)
        if torch.max(no_mines_proba) < self.out_of_boundary_treshold:
            # Allow to choose a non boundary box
            no_mines_proba += (~boundary)*1
        
        return self.valid_argmax2D(no_mines_proba.to('cpu'), grid_view)
    
    def get_probability_map(self):
        return self.last_map