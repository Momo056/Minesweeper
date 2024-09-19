from numpy import ndarray
from Players.Player_Interface import Player_Interface


class Tensor_Based_Player(Player_Interface):
    def action(self, grid: ndarray, grid_view: ndarray) -> tuple[int, int]:
        raise NotImplementedError()
