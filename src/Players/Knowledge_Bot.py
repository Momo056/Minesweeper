from numpy import ndarray
from src.Grid_Knowledge import MINE, NUMBER, Grid_Knowledge
from src.Players.Player_Interface import Player_Interface

import numpy as np

class Knowledge_Bot(Player_Interface):
    def __init__(self) -> None:
        super().__init__()
        self.grid_knowledge = Grid_Knowledge()

    def action(self, grid: ndarray, grid_view: ndarray) -> tuple[int, int] | None:
        # Forward information
        self.grid_knowledge.analyze(grid, grid_view)

        next_actions = np.argwhere(self.grid_knowledge.knowledge == NUMBER)

        if len(next_actions) == 0:
            # No action found
            return None
        
        return next_actions[0]
    
    def get_known_mines(self) -> ndarray[bool]:
        return self.grid_knowledge.knowledge == MINE
    

