import multiprocessing
import os
import time
from src.Game import Game
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.UI.GUI_Bot_Inputs import GUI_Bot_Inputs
from src.UI.No_UI import No_UI
from src.Grid import Grid

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import beta as beta_law
import matplotlib.pyplot as plt

import torch
from random import sample

from models.Game_Tensor_Interface import Game_Tensor_Interface

def gather_data(n_game:int, grid_size: int, mine_percent: int):
    tensor_list = []
    mines_list =[]
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
    return data, mines_tensor

def gather_data_one_arg(x):
    return gather_data(*x)

def main():
    grid_size = 12
    mine_percent = 0.22
    n_game = 10000


    # Show the number of CPUs available
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")

    # List of input values
    input_values = [(n_game, grid_size, mine_percent) for i in range(num_cpus)]
    parrallel_function = gather_data_one_arg

    start_t = time.perf_counter()
    with multiprocessing.Pool() as pool:
        all_res = pool.map(parrallel_function, input_values)
    # all_res = [parrallel_function(x) for x in input_values]
    end_t = time.perf_counter()
    print(f'Done in {(end_t-start_t):.2f}')

    tensor_representations = torch.cat([x for x, y in all_res])
    mines_tensors = torch.cat([y for x, y in all_res])

    # Save
    save_name = f'dataset/lose_bot/12x12_{len(tensor_representations)}.pt'
    while os.path.exists(save_name): # Change name until we find a new name
        save_name = save_name.replace('.pt', '_2.pt')
    torch.save([tensor_representations, mines_tensors], save_name)

if __name__ == "__main__":
    main()
