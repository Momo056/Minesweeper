from random import randint
from src.Players.Player_Interface import Player_Interface
import numpy as np


class No_Action_Player(Player_Interface):
    def action(self, grid: np.ndarray, grid_view: np.ndarray):
        return None
    