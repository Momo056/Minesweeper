from numpy import ndarray
from src.Constrains_Graph import Constrains_Graph
from src.Grid_Knowledge import MINE, NUMBER, UNCOVERED, Grid_Knowledge
import numpy as np


class Grid_Probability:
    def __init__(self, brut_force_limit: int=25) -> None:
        self.grid_knowledge = Grid_Knowledge()
        self.constraint_graph = Constrains_Graph()
        self.brut_force_limit = brut_force_limit

    def analyze(self, grid: ndarray, grid_view: ndarray) -> tuple[int, int]:
        # Reduce inforation to process
        self.grid_knowledge.analyze(grid, grid_view)

        # Compute graph
        self.constraint_graph.analyze(self.grid_knowledge)

        if len(self.constraint_graph.ordered_left) < self.brut_force_limit: # Only compute the real probability if there is a liited number of unknown boxes
            # Get all solutions
            solutions = self.constraint_graph.solve_matrix_form()
            box_probabilities = np.mean(solutions, axis=0)

            # Convert to array format
            self.mine_probability_map = -np.ones(grid.shape, dtype=float)
            self.mine_probability_map[self.grid_knowledge.knowledge == MINE] = 1
            self.mine_probability_map[self.grid_knowledge.knowledge == UNCOVERED] = 0
            self.mine_probability_map[self.grid_knowledge.knowledge == NUMBER] = 0

            for i, p in enumerate(box_probabilities):
                self.mine_probability_map[*self.constraint_graph.ordered_left[i]] = p
        else:
            self.mine_probability_map = None