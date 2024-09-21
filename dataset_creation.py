from argparse import ArgumentParser
import multiprocessing
import os
import time

from tqdm import tqdm
from itertools import product

from src.Game import Game
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.Grid import Grid

import numpy as np
import torch

from models.Game_Tensor_Interface import Game_Tensor_Interface


def gather_data(n_game: int, grid_size: int, mine_percent: float, save_path: str):
    tensor_list = []
    mines_list = []
    tensor_interface = Game_Tensor_Interface()

    for i in range(n_game):
        # Grid initialization
        grid = Grid(grid_size, grid_size, mine_percent)
        game = Game(grid)
        bot = Minesweeper_bot()
        game.action(*bot.action(*game.visible_grid()))

        # Play until we are about to make a mistake (or win)
        next_action = bot.action(*game.visible_grid())
        while not grid.mines[*next_action] and not game.is_ended():
            game.action(*next_action)
            if not game.is_ended():
                next_action = bot.action(*game.visible_grid())

        if grid.mines[*next_action]:
            tensor_list.append(tensor_interface.to_tensor(*game.visible_grid()))
            mines_list.append(grid.mines)

    data = torch.stack(tensor_list).type(torch.uint8)
    mines_tensor = torch.tensor(np.array(mines_list))

    torch.save([data, mines_tensor], save_path.replace('.pt', f'_l-{len(data)}.pt'))
    return


def gather_data_one_arg(x):
    return gather_data(*x)


def main():
    parser = ArgumentParser(description="Generate Minesweeper boards for training a model.")

    # Add arguments
    parser.add_argument('--grid_sizes', type=int, nargs='+', default=[12], help="List of grid sizes (default: [12])")
    parser.add_argument('--mine_percents', type=int, nargs='+', default=[22], help="List of mine percentages (default: [22])")
    parser.add_argument('--n_game', type=int, default=10000, help="Number of games to generate (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=1000, help="Number of games per batch (default: 1000)")
    parser.add_argument('--cpu', type=int, default=4, help="Number of CPUs to use")

    args = parser.parse_args()

    grid_sizes = args.grid_sizes
    mine_percents = [mp / 100.0 for mp in args.mine_percents]  # Convert to actual percentages
    n_game = args.n_game
    batch_size = args.batch_size

    # Create the dataset folder structure for each combination
    base_dataset_path = 'dataset'
    combinations = list(product(grid_sizes, mine_percents))
    
    input_values = []
    
    # Create folder structure and gather input values for each combination
    for grid_size, mine_percent in combinations:
        dataset_name = f'{grid_size}x{grid_size}_m{int(mine_percent * 100)}'
        dataset_path = os.path.join(base_dataset_path, dataset_name)

        if os.path.exists(dataset_path):
            index_offset = len(os.listdir(dataset_path))
        else:
            index_offset = 0
            os.makedirs(dataset_path, exist_ok=False)

        # Calculate the number of batches
        num_batches = n_game // batch_size
        remainder = n_game % batch_size

        # Create the list of arguments for each batch for this combination
        for i in range(num_batches):
            input_values.append((batch_size, grid_size, mine_percent, os.path.join(dataset_path, f"batch_{i+index_offset}.pt")))

        # Add the remainder as an extra batch if necessary
        if remainder > 0:
            input_values.append((remainder, grid_size, mine_percent, os.path.join(dataset_path, f"batch_{num_batches}.pt")))

    # Stratify the batches by round-robin scheduling
    stratified_batches = []
    max_batches = max(n_game // batch_size + (1 if n_game % batch_size > 0 else 0) for grid_size, mine_percent in combinations)

    for i in range(max_batches):
        for j in range(i, len(input_values), max_batches):
            stratified_batches.append(input_values[j])

    # Run the function in parallel using multiprocessing
    start_t = time.perf_counter()
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    
    with multiprocessing.Pool(min(num_cpus, args.cpu)) as pool:
        # Use tqdm to display a progress bar
        for _ in tqdm(pool.imap_unordered(gather_data_one_arg, stratified_batches), total=len(stratified_batches)):
            pass
    end_t = time.perf_counter()

    print(f"Done in {(end_t - start_t):.2f} seconds")



if __name__ == "__main__":
    main()
