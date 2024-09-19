from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def view_grid(
    grid_values: ndarray,
    grid_view: ndarray | None = None,
    mines: ndarray | None = None,
    fig: Figure = None,
    ax: Axes = None,
):
    if grid_view is None:
        grid_view = np.ones_like(grid_values).astype(np.bool_)

    if mines is None:
        mines = np.zeros_like(grid_values).astype(np.bool_)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    cmap_values = [
        "lightgray",
        "blue",
        "green",
        "red",
        "darkblue",
        "darkred",
        "cyan",
        "black",
        "darkgrey",
        "orange",
    ]
    for (j, i), label in np.ndenumerate(grid_values):
        if label != 0 and grid_view[j, i] and not mines[j, i]:
            ax.text(
                i,
                j,
                label,
                ha="center",
                va="center",
                color=cmap_values[label],
                fontweight="bold",
            )

    # Grid
    ax.grid(which="both", color="black", linestyle="-", linewidth=1)
    ax.set_xticks(np.arange(-0.5, grid_values.shape[1], 1), [])
    ax.set_yticks(np.arange(-0.5, grid_values.shape[0], 1), [])

    cmap_view = ListedColormap(["lightgray", "darkgray", "red", "darkred"])
    plot_array = np.zeros_like(grid_values) + 1 * (~grid_view) + 2 * mines
    # plot_array = plot_array*(~mines) + mines*3
    ax.imshow(plot_array, cmap=cmap_view, vmin=0, vmax=3)
