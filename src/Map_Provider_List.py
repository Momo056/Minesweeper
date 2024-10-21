from typing import Any, Callable

from numpy import ndarray


class Map_Provider_List:
    def __init__(self, probability_map_providers: list[Callable[[ndarray, ndarray], ndarray|None]]) -> None:
        self.probability_map_providers = probability_map_providers

    def __call__(self, grid: ndarray, grid_view: ndarray) -> ndarray | None:
        # Return first successfull estimation
        for provider in self.probability_map_providers:
            result = provider(grid, grid_view)
            if result is not None:
                return result   
        return None