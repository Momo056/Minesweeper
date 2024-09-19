from tkinter import Tk, Frame, Button

from .Game_Parameters import Game_Parameters
from src.Game import Game
from src.Grid import Grid
from src.UI.GUI_User_Inputs import GUI_User_Inputs


class App:
    def __init__(self, master: Tk) -> None:
        self.master = master
        self.game_parameters = Game_Parameters(10, 0.15)
        self.menu_frame = None  # Holds the menu buttons

    def start_game(self, game_parameters: Game_Parameters | None = None):
        if game_parameters is None:
            game_parameters = self.game_parameters
        grid = Grid(
            game_parameters.grid_size,
            game_parameters.grid_size,
            game_parameters.mine_percent,
        )
        game = Game(grid)
        gui = GUI_User_Inputs(self.master)
        return gui.start(game)

    def modify_parameters(self):
        # Function to modify the game parameters
        # You can implement this to open a new window for parameter modifications
        print("Modify parameters window opened (to be implemented)")

    def show_menu(self):
        # If the menu frame exists, destroy it to refresh the UI
        if self.menu_frame is not None:
            self.menu_frame.destroy()

        self.menu_frame = Frame(self.master)
        self.menu_frame.pack()

        start_button = Button(
            self.menu_frame, text="Start Game", command=self.start_game
        )
        start_button.pack(pady=10)

        modify_button = Button(
            self.menu_frame, text="Modify Parameters", command=self.modify_parameters
        )
        modify_button.pack(pady=10)

    def menu(self):
        self.show_menu()
