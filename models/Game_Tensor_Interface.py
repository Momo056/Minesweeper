from typing import Any
from numpy import ndarray
from torch import Tensor
import torch
from torch.nn.functional import one_hot
import numpy as np

from visualisation import view_grid


class Game_Tensor_Interface:
    def __init__(self, prepare_for_padding:bool=False) -> None:
        self.num_classes = 11 if prepare_for_padding else 10

    def to_tensor(self, grid: ndarray, grid_view: ndarray) -> Tensor:
        ## Input
        # [0, 8] : box value
        # {9} : unkown value
        # bomb value is unnecessary

        ## Output
        # One hot representation
        # {10} : For optinal pading values
        t = torch.tensor(grid + (~grid_view).astype(np.uint8)*9, dtype=torch.int64)

        return torch.permute(one_hot(t, self.num_classes).type(torch.float32), (-1, 0, 1))
    
    def to_grid(self, tensor_representation: Tensor) -> tuple[ndarray, ndarray]:
        # TODO
        pass
    
    def visible_grid(tensor_representation: Tensor):
        grid_view = tensor_representation[9]
        grid_values = torch.sum(tensor_representation[:9] * torch.arange(9, device=tensor_representation.device).view((9, 1, 1)), dim=0).type(torch.uint8)
        return np.array(grid_values).astype(np.uint8), ~(np.array(grid_view).astype(np.bool_))

    def view_grid_tensor(tensor_representation: Tensor, mines: Tensor|None = None, view_grid_kwargs:dict[str, Any]={}) -> None:
        return view_grid(*Game_Tensor_Interface.visible_grid(tensor_representation), mines=None if mines is None else np.array(mines), **view_grid_kwargs)
    
    def unknown_boundaries(tensor_representation: Tensor):
        return torch.logical_and(
            torch.conv2d(
                1-tensor_representation[:, -1::],
                torch.ones((1, 1, 3, 3), device=tensor_representation.device),
                padding=(1, 1)
            ) > 0,
            tensor_representation[:, -1::]
        )[:, 0]