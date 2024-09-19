from itertools import product
from math import floor
import random
import numpy as np
from scipy.signal import convolve2d


class Grid:
    def __init__(self, n_row: int, n_col: int, bomb_percent: float) -> None:
        self.mines = np.zeros((n_row, n_col), dtype=bool)
        # Place bombs
        self.n_bomb = floor(n_row * n_col * bomb_percent)
        bomb_coordinates = random.sample(
            list(product(range(n_row), range(n_col))), self.n_bomb
        )
        for x, y in bomb_coordinates:
            self.mines[x, y] = True

        # Compute the adjacents bombs
        self.update()

    def move_to_empty(self, x: int, y: int, depth: int = 0, max_depth: int = 10) -> int:
        # Find a good spot to move
        values = np.copy(self.grid)
        values[self.mines] = np.max(self.grid) + 1
        min_value = np.min(values)
        if values[x, y] == min_value or depth > max_depth:
            return values[x, y]

        possible_destinations = np.argwhere(values == min_value)
        # destination: np.ndarray = choices(possible_destinations, k=1)[0]
        destination = possible_destinations[len(possible_destinations) // 2]

        # Slide the grid
        self.slide_grid(*(np.array([x, y]) - destination))
        # On edge cases, we might still be not on the lowest possible grid value
        # We itrate until it is stable
        return self.move_to_empty(x, y, depth + 1)

    def slide_grid(self, delta_x: int, delta_y: int):
        self.mines = np.roll(
            np.roll(
                self.mines,
                delta_x,
                axis=0,
            ),
            delta_y,
            axis=1,
        )

        self.update()

    def update(self):
        # Update the number of adjacents if the mines array have been changed
        self.grid = convolve2d(self.mines, np.ones((3, 3)), mode="same").astype(
            np.uint8
        )

    def grid_shape(self):
        return self.mines.shape

    def discover(self, x: int, y: int):
        if self.grid[x, y] > 0:  # Cover the case of a mine
            result = np.zeros_like(self.mines)
            result[x, y] = True
            return result

        # Expand the area
        return self.expand(x, y)

    def expand(self, x: int, y: int):
        result = np.zeros_like(self.mines)
        result[x, y] = True
        last_result = np.zeros_like(self.mines)
        while np.any(last_result != result):  # Check if stabilized
            last_result = result

            # Restrict
            result = np.logical_and(
                result,
                self.grid == 0,
            )

            # Expand
            result = convolve2d(result, np.ones((3, 3)), mode="same").astype(bool)

        return result
