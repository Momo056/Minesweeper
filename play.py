import argparse
from tkinter import Tk

from numpy import ndarray
import numpy as np
from src.Players.No_Action_Player import No_Action_Player
from models.Game_Tensor_Interface import Game_Tensor_Interface
from src.UI.GUI_Hint_Map import GUI_Hint_Map
from src.Players import Player_Interface
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.UI.GUI_Hint_Grid import GUI_Hint_Grid
from src.Game import Game
from src.Grid import Grid
from src.UI.GUI_User_Inputs import GUI_User_Inputs
from src.App.App import App

def parse_arguments():
    parser = argparse.ArgumentParser(description="Options for Minesweeper game.")
    
    # Taille de la grille: soit un entier n pour une grille carrée, soit deux entiers w et h pour une grille rectangulaire
    parser.add_argument('--grid_size', type=int, nargs='+', default=[10, 10], 
                        help="Taille de la grille : soit un entier pour une grille carrée, soit deux pour une grille rectangulaire (par défaut : 10 10).")
    
    # Pourcentage de mines dans la grille
    parser.add_argument('--mine_percent', type=float, default=0.15, 
                        help="Pourcentage de mines dans la grille (par défaut : 0.15).")
    

    # Chemin vers le fichier définissant la structure du modèle PyTorch
    parser.add_argument('--model_type', type=str, default=None, 
                        help="Chemin vers le fichier qui définit la structure du modèle PyTorch (par défaut : None).")
    
    # Chemin vers les poids du modèle PyTorch
    parser.add_argument('--model_parameters', type=str, default=None, 
                        help="Chemin vers le fichier contenant les poids du modèle PyTorch (par défaut : None).")
      
    return parser.parse_args()


def get_probability_model(model_type: str, model_parameters: str) -> Player_Interface:
    print('Loading torch')
    from models.Model_Dict import MODEL_PROVIDER_DICT
    from models.Model_Based_Player import Model_Based_Player
    import torch
    print('torch loaded')

    model = MODEL_PROVIDER_DICT[model_type]()
    state_dict = torch.load(model_parameters)
    model.load_state_dict(state_dict)
    model.to('cuda')
    tensor_interface = Game_Tensor_Interface()

    def map_provider(grid: ndarray, grid_view: ndarray) -> ndarray:
        tensor_representation = tensor_interface.to_tensor(grid, grid_view).type(torch.float32).to('cuda')
        model.eval()
        with torch.no_grad():
            model_output = model(tensor_representation.reshape((1, *tensor_representation.shape)))[0]

        no_mines_proba = torch.exp(model_output[0]).to('cpu')
        
        return np.array(no_mines_proba)

    return map_provider


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Parameters
    if len(args.grid_size) == 1:
        grid_size = (args.grid_size[0], args.grid_size[0])
    else:
        grid_size = (args.grid_size[0], args.grid_size[1])
    
    mine_percent = args.mine_percent
    
    # Instanciate the bot
    minesweeper_bot = Minesweeper_bot(True, True, delegated_if_no_solution=No_Action_Player())  # Utilisation du modèle si nécessaire (à adapter si besoin)

    
    master = Tk()

    if args.model_type is not None:
        # Load model
        map_provider = get_probability_model(args.model_type, args.model_parameters)
        gui = GUI_Hint_Map(minesweeper_bot, map_provider, master)
    else:
        gui = GUI_Hint_Grid(minesweeper_bot, master)
    # minesweeper_bot = delegate_model

    # Main code
    grid = Grid(*grid_size, mine_percent)
    game = Game(grid)
    gui.start(game)
