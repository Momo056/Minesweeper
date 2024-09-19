from random import randint
from src.Players.Player_Interface import Player_Interface
import numpy as np


class Random_Player(Player_Interface):
    def action(self, grid: np.ndarray, grid_view: np.ndarray):
        return randint(0, grid.shape[0] - 1), randint(0, grid.shape[1] - 1)
