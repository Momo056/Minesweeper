from itertools import product
from random import choice, randint
from numpy import ndarray
from src.Includes import box
from src.Players.Player_Interface import Player_Interface
import numpy as np


class Minesweeper_bot(Player_Interface):
    def __init__(
        self,
        random_gambit: bool = True,
        random_first_move: bool = False,
        delegated_if_no_solution: Player_Interface | None = None,
    ) -> None:
        super().__init__()
        self.random_gambit = random_gambit
        self.random_first_move = random_first_move
        self.delegated_if_no_solution = delegated_if_no_solution

    def run_1_box_analysis(self) -> tuple[int, int] | None:
        while len(self.to_inspect) > 0:
            x, y = self.to_inspect.pop(0)
            solution = self.inspect(x, y)
            if solution is not None:
                return solution
        return None

    def run_2_boxes_analysis(self) -> tuple[int, int] | None:
        while len(self.to_cross_inspect) > 0:
            p1, p2 = self.to_cross_inspect.pop(0)
            solution = self.cross_inspect(p1, p2)
            if solution is not None:
                return solution

            # Might need to inspect boxes
            solution = self.run_1_box_analysis()
            if solution is not None:
                return solution
        return None

    def determine_as_mines(self, *p_list: list[box]):
        for i, j in p_list:
            if self.unkown_region[i, j] and not self.known_mines[i, j]:
                self.known_mines[i, j] = True
                # Might give information to neighbors of the mine
                self.to_inspect += [
                    n for n in self.all_neighbors(i, j) if self.grid_view[*n]
                ]
                # TODO : Propagate information for cross checking

    def action(self, grid: ndarray, grid_view: ndarray) -> tuple[int, int]:
        self.grid = grid
        self.grid_view = grid_view

        # Only make sense on the ~grid_view array
        self.unkown_region = ~grid_view
        self.known_mines = np.zeros_like(grid_view)
        # self.known_no_mines = np.copy(grid_view) # Useless, if we found a known_no_mines, we return it imediatly

        if np.all(~self.grid_view):
            # First action of the game
            if self.random_first_move:
                return randint(0, grid_view.shape[0] - 1), randint(
                    0, grid_view.shape[1] - 1
                )
            return grid_view.shape[0] // 2, grid_view.shape[1] // 2

        # One box rule on knowledge
        self.to_inspect = list(np.argwhere(np.logical_and(grid_view, grid > 0)))
        solution = self.run_1_box_analysis()
        if solution is not None:
            return solution

        # Two boxes rules of knowledge
        self.to_cross_inspect = list(
            np.argwhere(np.logical_and(grid_view, grid > 0))
        )  # TODO : Supress boxs without neighbours
        self.to_cross_inspect = [
            (p1, p2)
            for p1, p2 in product(self.to_cross_inspect, self.to_cross_inspect)
            if not np.all(p1 == p2)
        ]
        solution = self.run_2_boxes_analysis()
        if solution is not None:
            return solution

        # No action have been deduce, we sample a possible move where we do not know if there is a bomb
        if self.delegated_if_no_solution is not None:
            return self.delegated_if_no_solution.action(grid, grid_view)

        possible_actions = np.argwhere(
            np.logical_and(
                ~grid_view,
                ~self.known_mines,
            )
        )

        if self.random_gambit:
            return choice(possible_actions)
        return possible_actions[0]

    def all_neighbors(self, x: int, y: int):
        # List all neigbhors
        neighbors = [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if 0 <= x + dx
            and x + dx < self.known_mines.shape[0]
            and 0 <= y + dy
            and y + dy < self.known_mines.shape[1]
            and not (dx == 0 and dy == 0)
        ]
        return neighbors

    def inspect(self, x: int, y: int) -> tuple[int, int] | None:
        value = self.grid[x, y]

        # Case value == 0 isn't interestng because all the neighbors are values
        if self.grid_view[x, y] and value > 0:
            neighbors = self.all_neighbors(x, y)
            unknown_boxes = [(i, j) for i, j in neighbors if self.unkown_region[i, j]]
            if len(unknown_boxes) == value:
                # All bombs
                self.determine_as_mines(*neighbors)
            else:
                # Check if all the mines have been founded
                founded_mines = [(i, j) for i, j in neighbors if self.known_mines[i, j]]
                if len(founded_mines) == value:
                    # All remaining boxes are safe
                    for i, j in neighbors:
                        if self.unkown_region[i, j] and not self.known_mines[i, j]:
                            return i, j
                else:
                    # No decision can be made
                    pass

        # We didn't deduce
        return None

    def cross_inspect(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> tuple[int, int] | None:
        if abs(p1[0] - p2[0]) > 2 or abs(p1[1] - p2[1]) > 2:
            # No intersection
            return None
        # Values agregation
        c1 = self.grid[*p1] - len(
            [(i, j) for i, j in self.all_neighbors(*p1) if self.known_mines[i, j]]
        )  # Remove knowned mine count
        c2 = self.grid[*p2] - len(
            [(i, j) for i, j in self.all_neighbors(*p2) if self.known_mines[i, j]]
        )  # Remove knowned mine count
        v1 = [
            (i, j)
            for i, j in self.all_neighbors(*p1)
            if self.unkown_region[i, j] and not self.known_mines[i, j]
        ]
        v2 = [
            (i, j)
            for i, j in self.all_neighbors(*p2)
            if self.unkown_region[i, j] and not self.known_mines[i, j]
        ]
        vi = [p for p in v1 if p in v2]

        # Inverted condition for else as a return
        if c2 != c1 - len(v1) + len(vi):
            return None

        # v2\vi has no mines
        v2_outer = [p for p in v2 if p not in vi]
        if len(v2_outer) > 0:
            return v2_outer[0]

        # T(v1\vi) = T(v1) - T(v2)
        # If T(v1\vi) == |v1\vi| -> Only mines
        v1_outer = [p for p in v1 if p not in vi]
        if c1 - c2 == len(v1_outer):
            # v1\vi is all mines
            self.determine_as_mines(*v1_outer)

        return None

    def get_known_mines(self) -> ndarray:
        return self.known_mines

    def get_probability_map(self):
        return self.delegated_if_no_solution.get_probability_map()
