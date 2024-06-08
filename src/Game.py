from math import prod
from tkinter import Grid
import numpy as np


class Game:
    def __init__(
            self,
            grid: Grid,
        ) -> None:
        self.grid = grid
        # What the player can see
        self.player_grid_view = np.zeros(grid.grid_shape(), dtype=bool)
    
    def action(self, x: int, y: int):# Discover new part (or not new) of the grid
        if np.sum(self.player_grid_view) == 0:
            # First action of the player, we move the grid such that the player input is on a empty cell
            self.grid.move_to_empty(x, y)

        self.player_grid_view = np.logical_or(
            self.player_grid_view,
            self.grid.discover(x, y),
        )
    
    def is_ended(self):
        result = self.result()
        return result is not None
    
    def result(self):
        # Discovered a mine
        if np.any(np.logical_and(self.player_grid_view, self.grid.mines)):
            return False
        
        # Discovered has many tile than free tiles
        return None if self.grid.n_bomb + np.sum(self.player_grid_view) < prod(self.grid.grid_shape()) else True
    
    def visible_grid(self):
        return self.grid.grid*(self.player_grid_view), self.player_grid_view