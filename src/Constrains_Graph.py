from dataclasses import dataclass
from src.Grid_Knowledge import UNCOVERED, UNKNOWN, Grid_Knowledge, MINE, Box
import numpy as np
from numba import jit

# JIT-compiled function for binary array generation
@jit(nopython=True)
def generate_binary_array(n_bits):
    n_rows = 2 ** n_bits
    result = np.zeros((n_rows, n_bits), dtype=np.int8)
    
    for i in range(n_rows):
        for j in range(n_bits):
            # Fill each row with the binary representation of i
            result[i, n_bits - j - 1] = (i >> j) & 1
            
    return result
generate_binary_array(4) # Initialize the compilation
pass

@dataclass
class Right_Node:
    value: int
    neighbors: set[Box]

class Constrains_Graph:
    def __init__(self) -> None:
        # Bipartite graph
        # Unknown boxes on the left. Node value in {0, 1}
        # Uncovered boxes on the right. Node values in {1, 2, 3, 4, 5, 6, 7}
        # Edge represent the adjacency in Minesweeper
        # The sum of the left nodes connected to a right node should equal to the value of the right node
        
        self.right_nodes: dict[Box, Right_Node] = {}
        self.A = None
        self.b = None


    def analyze(self, grid_knowledge: Grid_Knowledge):
        print('Graph knowledge')
        print(grid_knowledge.knowledge)
        self.grid_knowledge = grid_knowledge
        possible_left = [tuple(b.tolist()) for b in np.argwhere(grid_knowledge.knowledge == UNKNOWN)]
        print('Should all be UNKNOWN (1)')
        print([grid_knowledge.knowledge[*l] for l in possible_left])
        print('Corresponding lefts')
        print(possible_left)
        print()

        # Reset right nodes
        self.right_nodes = {}

        # Construct the graph
        for left in possible_left:
            left_neighbors = grid_knowledge.all_neighbors(*left)

            for left_n in left_neighbors:
                if grid_knowledge.knowledge[*left_n] == UNCOVERED:
                    # For every neighbor of a left that has a known value
                    self.append_edge(left, left_n)

        # Matrix form
        self.compute_matrix_form()

    def append_edge(self, left: Box, right: Box):
        if right not in self.right_nodes.keys():
            self.right_nodes[right] = Right_Node(self.compute_right_value(right), set())

        self.right_nodes[right].neighbors.add(left)
        print(f'Add : {left} <- {right}')

    def compute_right_value(self, right: Box):
        right_neighbors = self.grid_knowledge.all_neighbors(*right)

        base_value = self.grid_knowledge.grid[*right]

        mine_neighbors = [n for n in right_neighbors if self.grid_knowledge.knowledge[*n] == MINE]

        return base_value - len(mine_neighbors)
    
    def compute_matrix_form(self):
        self.ordered_right = sorted(list(self.right_nodes.keys()))

        self.ordered_left = sorted(list({
            b
            for r_node in self.right_nodes.values()
            for b in r_node.neighbors
        }))

        self.index_right: dict[Box, int] = {b:i for i, b in enumerate(self.ordered_right)}
        self.index_left: dict[Box, int] = {b:i for i, b in enumerate(self.ordered_left)}

        self.A = np.zeros((len(self.ordered_right), len(self.ordered_left)), dtype=int)
        self.b = np.zeros((len(self.ordered_right)), dtype=int)

        for right, r_node in self.right_nodes.items():
            self.b[self.index_right[right]] = r_node.value
            for left in r_node.neighbors:
                self.A[self.index_right[right], self.index_left[left]] = 1

    def solve_matrix_form(self):
        if self.A is None:
            self.compute_matrix_form()

        x = generate_binary_array(self.A.shape[-1])
        y = x @ self.A.T
        valid_y = np.all(self.b.reshape(1, -1) == y, axis=-1)
        return x[valid_y]


