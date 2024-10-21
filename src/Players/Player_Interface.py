import numpy as np


class Player_Interface:
    def action(self, grid: np.ndarray, grid_view: np.ndarray) -> tuple[int, int] | None:
        pass

    def get_known_mines(self) -> np.ndarray[bool]:
        pass
