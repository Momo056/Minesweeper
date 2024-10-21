import argparse
from tkinter import Tk

from numpy import ndarray
import numpy as np
from src.Brut_Force_Map_Provider import Brut_Force_Map_Provider
from src.Grid_Probability import Grid_Probability
from src.Map_Provider_List import Map_Provider_List
from src.Players.Knowledge_Bot import Knowledge_Bot
from src.Players.No_Action_Player import No_Action_Player
from src.UI.GUI_Hint_Map import GUI_Hint_Map
from src.Players import Player_Interface
from src.Players.Minesweeper_bot import Minesweeper_bot
from src.UI.GUI_Hint_Grid import GUI_Hint_Grid
from src.Game import Game
from src.Grid import Grid


def parse_arguments():
    parser = argparse.ArgumentParser(description="Options for Minesweeper game.")

    # Taille de la grille: soit un entier n pour une grille carrée, soit deux entiers w et h pour une grille rectangulaire
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs="+",
        default=[10, 10],
        help="Taille de la grille : soit un entier pour une grille carrée, soit deux pour une grille rectangulaire (par défaut : 10 10).",
    )

    # Pourcentage de mines dans la grille
    parser.add_argument(
        "--mine_percent",
        type=float,
        default=0.15,
        help="Pourcentage de mines dans la grille (par défaut : 0.15).",
    )

    # Chemin vers le fichier définissant la structure du modèle PyTorch
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Chemin vers le fichier qui définit la structure du modèle PyTorch (par défaut : None).",
    )

    # Chemin vers les poids du modèle PyTorch
    parser.add_argument(
        "--model_parameters",
        type=str,
        default=None,
        help="Chemin vers le fichier contenant les poids du modèle PyTorch (par défaut : None).",
    )

    # Use brut force probability estimator
    parser.add_argument(
        "--use_bruteforce",
        action="store_true",
        help="Active l'estimateur de probabilité par force brute. Si non spécifié, l'estimateur n'est pas utilisé.",
    )

    # Limit for the brute force method
    parser.add_argument(
        "--bruteforce_limit",
        type=int,
        default=25,
        help="Limite pour la méthode de force brute (doit être un entier positif, par défaut : 25).",
    )

    return parser.parse_args()


def get_probability_model(model_type: str, model_parameters: str):
    print("Loading torch")
    from Lightning.Model_provider import MODEL_PROVIDER_DICT
    import torch
    from models.Game_Tensor_Interface import Game_Tensor_Interface

    print("torch loaded")

    model = MODEL_PROVIDER_DICT[model_type]()
    checkpoint = torch.load(model_parameters)
    model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda")
    tensor_interface = Game_Tensor_Interface()

    def map_provider(grid: ndarray, grid_view: ndarray) -> ndarray:
        tensor_representation = (
            tensor_interface.to_tensor(grid, grid_view).type(torch.float32).to("cuda")
        )
        model.eval()
        with torch.no_grad():
            model_output = model(
                tensor_representation.reshape((1, *tensor_representation.shape))
            )[0]

        no_mines_proba = torch.exp(model_output[0]).to("cpu")

        return np.array(no_mines_proba)

    return map_provider


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Parameters
    if len(args.grid_size) == 1:
        grid_size = (args.grid_size[0], args.grid_size[0])
    else:
        grid_size = (args.grid_size[0], args.grid_size[1])

    mine_percent = args.mine_percent

    # Instanciate the bot
    # minesweeper_bot = Minesweeper_bot(
    #     True, True, delegated_if_no_solution=No_Action_Player()
    # )  # Utilisation du modèle si nécessaire (à adapter si besoin)
    minesweeper_bot = Knowledge_Bot()

    master = Tk()

    # Probability map provider building
    # Providers listed by priority
    # The first one to return a successfull resutl is used
    map_providers = []
    
    if args.use_bruteforce:
        def print_wrapper(func: Brut_Force_Map_Provider):
            def print_exec(*args):
                print('-'*100)
                result = func(*args)
                print()
                print('Knowledge')
                print(func.grid_probability.grid_knowledge.knowledge)
                print()
                print('Probabilities')
                if func.grid_probability.mine_probability_map is not None:
                    print(np.round(100*func.grid_probability.mine_probability_map))
                print()
                print('Constraints')
                print('Left')
                print(func.grid_probability.constraint_graph.ordered_left)
                print('Right')
                print(func.grid_probability.constraint_graph.ordered_right)
                return result
            return print_exec
        map_providers.append(
            print_wrapper(Brut_Force_Map_Provider(Grid_Probability(args.bruteforce_limit)))
        )

    if args.model_type is not None:
        # Load model
        map_providers.append(get_probability_model(args.model_type, args.model_parameters))

    map_provider = Map_Provider_List(map_providers)

    gui = GUI_Hint_Map(minesweeper_bot, map_provider, master)

    # Main code
    grid = Grid(*grid_size, mine_percent)
    game = Game(grid)
    gui.start(game)
