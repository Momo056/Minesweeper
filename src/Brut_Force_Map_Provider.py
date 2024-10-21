from typing import Any

from numpy import ndarray

from src.Grid_Probability import Grid_Probability


class Brut_Force_Map_Provider:
    def __init__(self, grid_probability: Grid_Probability) -> None:
        self.grid_probability = grid_probability

    def __call__(self, grid: ndarray, grid_view: ndarray) -> ndarray | None:
        self.grid_probability.analyze(grid, grid_view)
        if self.grid_probability.mine_probability_map is None:
            return None
        return 1 - self.grid_probability.mine_probability_map