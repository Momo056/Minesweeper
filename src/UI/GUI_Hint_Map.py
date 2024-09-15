import tkinter as tk
from typing import Any, Callable
import numpy as np
from src.Players.Player_Interface import Player_Interface
from src.Game import Game

class GUI_Hint_Map:
    def __init__(self, bot: Player_Interface, map_provider: Callable[[np.ndarray, np.ndarray], np.ndarray], master: tk.Tk | None=None):
        self.master = master if master is not None else tk.Tk()
        self.master.title("Minesweeper with Hints")
        self.master.geometry("1920x720")  # Adjusted to accommodate 3 grids

        self.bot = bot

        self.map_provider = map_provider

        # Initialize generated action variable
        self.generated_action = None
        self.game = None

    def start(self, game: Game):
        self.create_widgets(game)
        
        # Update grids of the minesweeper
        self.initialize_grid(game)
        self.initialize_hint_grid(game)
        self.initialize_probability_grid(game)    

        # Launch the app
        self.master.mainloop()

        return game.result()

    def create_widgets(self, game: Game):
        # Create a frame to contain all three grids side by side
        self.main_frame = tk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10)

        # Create player grid frame (left side)
        self.grid_frame = tk.Frame(self.main_frame)
        self.grid_frame.grid(row=0, column=0, padx=10)

        # Create hint grid frame (middle)
        self.hint_grid_frame = tk.Frame(self.main_frame)
        self.hint_grid_frame.grid(row=0, column=1, padx=10)

        # Create probability grid frame (right side)
        self.probability_grid_frame = tk.Frame(self.main_frame)
        self.probability_grid_frame.grid(row=0, column=2, padx=10)

        # Status bar at the bottom of the window
        self.status_bar = tk.Label(self.master, text="Total number of mines: Unknown", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="we")

        # Add new button for playing generated action
        self.generated_action_button = tk.Button(self.master, text="Play Generated Action", command=lambda: self.play_generated_action(game))
        self.generated_action_button.grid(row=2, column=0, pady=10)

    def play_generated_action(self, game: Game):
        if self.generated_action is not None:
            row, col = self.generated_action
            # Play the action and update the grid
            self.on_button_click(game, row, col)
        else:
            print("No action is generated yet.")

    def initialize_grid(self, game: Game, frame: tk.Frame=None):
        if frame is None:
            frame = self.grid_frame

        # Initialize flag storage
        self.flags = np.zeros_like(game.player_grid_view)

        # Create the buttons/grid for the player
        self.buttons: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(frame, width=2, height=1, command=lambda r=row, c=col: self.on_button_click(game, r, c))
                button.grid(row=row, column=col)
                button.bind("<Button-3>", lambda e, r=row, c=col: self.on_right_click(game, r, c))
                row_buttons.append(button)
            self.buttons.append(row_buttons)

        self.status_bar.config(text=f"Total number of mines: {game.grid.n_bomb}")

        self.update_grid(game)

    def initialize_hint_grid(self, game: Game, frame: tk.Frame=None):
        if frame is None:
            frame = self.hint_grid_frame

        # Create the buttons/grid for the hints
        self.hint_buttons: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(frame, width=2, height=1, state=tk.DISABLED)
                button.grid(row=row, column=col)
                row_buttons.append(button)
            self.hint_buttons.append(row_buttons)

        self.update_hint_grid(game)

    def initialize_probability_grid(self, game: Game, frame: tk.Frame=None):
        if frame is None:
            frame = self.probability_grid_frame

        # Create the buttons/grid for the probabilities
        self.probability_buttons: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(frame, width=2, height=1, state=tk.DISABLED)
                button.grid(row=row, column=col)
                row_buttons.append(button)
            self.probability_buttons.append(row_buttons)

        self.update_probability_grid(game)

    def on_button_click(self, game: Game, row: int, col: int):
        game.action(row, col)
        self.update_grid(game)
        self.update_hint_grid(game)
        self.update_probability_grid(game)

    def on_right_click(self, game: Game, row: int, col: int):
        self.flags[row, col] = not self.flags[row, col]
        self.update_grid(game)
        self.update_hint_grid(game)
        self.update_probability_grid(game)

    def update_grid(self, game: Game):
        # Update the player's grid view
        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.buttons[row][col]

                if not game.player_grid_view[row, col]:
                    if self.flags[row, col]:
                        text_button = 'F'
                        color = 'yellow'
                    else:
                        text_button = ''
                        color = 'gray'
                elif game.grid.mines[row, col]:
                    text_button = 'M'
                    color = 'red'
                else:
                    value = int(game.grid.grid[row, col])
                    text_button = str(value) if value != 0 else ''
                    color = 'lightgrey'

                button.config(text=text_button, bg=color)

    def update_hint_grid(self, game: Game):
        if game.is_ended():
            return
        
        hints = self.calculate_best_moves(game)
        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.hint_buttons[row][col]
                
                if game.player_grid_view[row, col]:
                    self.update_visible_button(button, int(game.grid.grid[row, col]))
                else:
                    self.update_hint_button(button, hints[row, col])

    def update_probability_grid(self, game: Game):
        probability_map = self.map_provider(*game.visible_grid())

        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.probability_buttons[row][col]
                
                if game.player_grid_view[row, col]:
                    self.update_visible_button(button, int(game.grid.grid[row, col]))
                else:
                    prob = None if probability_map is None else probability_map[row, col]
                    self.update_probability_button(button, prob)

    def update_visible_button(self, button: tk.Button, value: int):
        text_button = str(value) if value != 0 else ''
        color = 'lightgrey'
        button.config(text=text_button, bg=color)

    def update_hint_button(self, button: tk.Button, hint_value: Any):
        if hint_value == 1:
            text_button = 'V'  # Safe move
            color = 'green'
        elif hint_value == -1:
            text_button = 'X'  # Mine
            color = 'red'
        else:
            text_button = ''
            color = 'gray'
        button.config(text=text_button, bg=color)

    def update_probability_button(self, button: tk.Button, probability: float | None):
        if probability is not None:
            # Format the probability to two decimal places and ensure it starts with a dot (e.g., ".39" for 0.38856)
            text_button = f"{(1-probability):.2f}"[1:]  # Slice off the leading '0' to keep the format '.XX'
            color = self.get_color(probability)
        else:
            text_button = ''
            color = 'gray'
        button.config(text=text_button, bg=color)

    def get_color(self, value: float):
        value = max(0, min(1, value))
        normalized_value = value

        red = int((1 - normalized_value) * 255)
        green = int((1 - normalized_value) * 165 + normalized_value * 255)
        blue = int(normalized_value * 255)

        return f'#{red:02x}{green:02x}{blue:02x}'

    def calculate_best_moves(self, game: Game):
        hints = np.zeros_like(game.player_grid_view, dtype=np.int64)

        bot_action = self.bot.action(*game.visible_grid())
        if bot_action is not None:
            hints[*bot_action] = 1
            self.generated_action = bot_action
        
        try:
            known_mines = self.bot.get_known_mines()
            hints[known_mines] = -1
        except AttributeError:
            pass

        return hints
