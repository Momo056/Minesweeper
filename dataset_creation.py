from argparse import ArgumentParser
import multiprocessing
import os
import time

from tqdm import tqdm
from src.Game import Game
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.Grid import Grid

import numpy as np
import numpy as np

import torch

from models.Game_Tensor_Interface import Game_Tensor_Interface

def gather_data(n_game: int, grid_size: int, mine_percent: int, save_path: str):
    tensor_list = []
    mines_list = []
    tensor_interface = Game_Tensor_Interface()

    for i in range(n_game):
        # Grid initialisation
        grid = Grid(grid_size, grid_size, mine_percent)
        game = Game(grid)
        bot = Minesweeper_bot()
        game.action(*bot.action(*game.visible_grid()))

        # Play until we are about to do a mistake (or we win)
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
    parser = ArgumentParser(description="Générer des plateaux de Minesweeper pour entraîner un modèle.")

    # Ajout des arguments
    parser.add_argument('--grid_size', type=int, default=12, help="Taille de la grille (par défaut: 12)")
    parser.add_argument('--mine_percent', type=int, default=22, help="Pourcentage de mines dans la grille, sous forme entière (par défaut: 22 pour 22%)")
    parser.add_argument('--n_game', type=int, default=10000, help="Nombre de parties à générer (par défaut: 10000)")
    parser.add_argument('--batch_size', type=int, default=1000, help="Nombre de parties par batch (par défaut: 1000)")
    parser.add_argument('--cpu', type=int, default=4, help="Nombre de CPU utilisés")

    args = parser.parse_args()

    # Assignation des valeurs des arguments
    grid_size = args.grid_size
    mine_percent = args.mine_percent / 100.0  # Conversion en pourcentage réel
    n_game = args.n_game
    batch_size = args.batch_size

    # Placeholder for the batched data
    dataset_name = f'{grid_size}x{grid_size}_m{args.mine_percent}'
    dataset_path = os.path.join('dataset', dataset_name)
    if os.path.exists(dataset_path):
        index_offset = len(os.listdir(dataset_path))
    else:
        index_offset = 0
        os.makedirs(dataset_path, exist_ok=False)

    # Show the number of CPUs available
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")

    # Calculate the number of batches
    num_batches = n_game // batch_size
    remainder = n_game % batch_size

    # Create the list of arguments for each batch
    input_values = [(batch_size, grid_size, mine_percent, os.path.join(dataset_path, f"batch_{i+index_offset}.pt"))
                    for i in range(num_batches)]

    # Add the remainder as an extra batch if necessary
    if remainder > 0:
        input_values.append((remainder, grid_size, mine_percent, os.path.join(dataset_path, f"batch_{num_batches}.pt")))

    # Run the function in parallel using multiprocessing
    start_t = time.perf_counter()
    with multiprocessing.Pool(min(num_cpus, args.cpu)) as pool:
        # Use tqdm to display a progress bar
        for _ in tqdm(pool.imap_unordered(gather_data_one_arg, input_values), total=len(input_values)):
            pass
    end_t = time.perf_counter()

    print(f"Done in {(end_t - start_t):.2f} seconds")

if __name__ == "__main__":
    main()


