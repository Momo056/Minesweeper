import tkinter as tk
import numpy as np
from src.Game import Game

class GUI_User_Inputs:
    def __init__(self, master: tk.Tk | None=None):
        self.master = master if master is not None else tk.Tk()
        self.master.title("Minesweeper")
        self.master.geometry("1440x720")

        # Create the structure to place the game grid
        self.create_widgets()

    def start(self, game: Game):
        # Update grid of the minesweeper
        self.initialize_grid(game)

        # Lunch the app
        self.master.mainloop()

        return game.result()        

    def create_widgets(self):
        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(pady=10)

        self.status_bar = tk.Label(self.master, text=f"Total number of mines : Unknow", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # self.initialize_grid()
        # self.generate_mines(10)

    def initialize_grid(self, game: Game):
        # Initialize flag storage
        self.flags = np.zeros_like(game.player_grid_view)

        # Create the buttons/grid
        self.buttons: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(self.grid_frame, width=2, height=1, command=lambda r=row, c=col: self.on_button_click(game, r, c))
                button.grid(row=row, column=col)
                button.bind("<Button-3>", lambda e, r=row, c=col: self.on_right_click(game, r, c))
                row_buttons.append(button)
            self.buttons.append(row_buttons)

        # Upadte the label
        self.status_bar.config(text=f"Total number of mines : {game.grid.n_bomb}")

        self.update_grid(game)

    def on_button_click(self, game: Game, row: int, col: int):
        game.action(row, col)

        self.update_grid(game)

    def on_right_click(self, game: Game, row: int, col: int):
        self.flags[row, col] = not self.flags[row, col]
        self.update_grid(game)

    def update_grid(self, game: Game):
        # Place the numbers on the buttons
        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.buttons[row][col]

                if not game.player_grid_view[row, col]:
                    # Covered boxes
                    if self.flags[row, col]:
                        text_button = 'F'
                        color='yellow'
                    elif game.grid.mines[row, col] and game.is_ended():
                        # Show the mines at the end of the game
                        if game.result():
                            text_button = 'F'
                            color='yellow'
                        else:
                            text_button = 'M'
                            color='orange'
                    else:
                        text_button = ''
                        color='gray'
                elif game.grid.mines[row, col]:
                    text_button = 'M'
                    color='red'
                else:
                    value = int(game.grid.grid[row, col])
                    text_button = str(value) if value != 0 else ''
                    color='lightgrey'

                button.config(text=text_button, bg=color)
