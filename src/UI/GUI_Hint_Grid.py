import tkinter as tk
from typing import Any
import numpy as np
from src.Players.Player_Interface import Player_Interface
from src.Game import Game


class GUI_Hint_Grid:
    def __init__(self, bot: Player_Interface, master: tk.Tk | None = None):
        self.master = master if master is not None else tk.Tk()
        self.master.title("Minesweeper with Hints")
        self.master.geometry("1440x720")

        self.bot = bot
        # Create the structure to place both game grids (player and hint)
        self.create_widgets()

        self.old_probability_map = None

    def start(self, game: Game):
        # Update grids of the minesweeper
        self.initialize_grid(game)
        self.initialize_hint_grid(game)

        # Launch the app
        self.master.mainloop()

        return game.result()

    def create_widgets(self):
        # Create a frame to contain both grids side by side
        self.main_frame = tk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10)

        # Create player grid frame (left side)
        self.grid_frame = tk.Frame(self.main_frame)
        self.grid_frame.grid(row=0, column=0, padx=10)

        # Create hint grid frame (right side)
        self.hint_grid_frame = tk.Frame(self.main_frame)
        self.hint_grid_frame.grid(row=0, column=1, padx=10)

        # Status bar at the bottom of the window
        self.status_bar = tk.Label(
            self.master,
            text="Total number of mines: Unknown",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        self.status_bar.grid(row=1, column=0, sticky="we")

    def initialize_grid(self, game: Game, frame: tk.Frame = None):
        if frame is None:
            frame = self.grid_frame

        # Initialize flag storage
        self.flags = np.zeros_like(game.player_grid_view)

        # Create the buttons/grid for the player
        self.buttons: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(
                    frame,
                    width=2,
                    height=1,
                    command=lambda r=row, c=col: self.on_button_click(game, r, c),
                )
                button.grid(row=row, column=col)
                button.bind(
                    "<Button-3>",
                    lambda e, r=row, c=col: self.on_right_click(game, r, c),
                )
                row_buttons.append(button)
            self.buttons.append(row_buttons)

        self.status_bar.config(text=f"Total number of mines: {game.grid.n_bomb}")

        self.update_grid(game)

    def initialize_hint_grid(self, game: Game, frame: tk.Frame = None):
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

    def on_button_click(self, game: Game, row: int, col: int):
        game.action(row, col)
        self.update_grid(game)
        self.update_hint_grid(game)

    def on_right_click(self, game: Game, row: int, col: int):
        self.flags[row, col] = not self.flags[row, col]
        self.update_grid(game)
        self.update_hint_grid(game)

    def update_grid(self, game: Game):
        # Update the player's grid view
        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.buttons[row][col]

                if not game.player_grid_view[row, col]:
                    if self.flags[row, col]:
                        text_button = "F"
                        color = "yellow"
                    else:
                        text_button = ""
                        color = "gray"
                elif game.grid.mines[row, col]:
                    text_button = "M"
                    color = "red"
                else:
                    value = int(game.grid.grid[row, col])
                    text_button = str(value) if value != 0 else ""
                    color = "lightgrey"

                button.config(text=text_button, bg=color)

    def update_hint_grid(self, game: Game):
        if game.is_ended():
            return

        # Update the hint grid with best moves suggested
        hints = self.calculate_best_moves(
            game
        )  # This is a placeholder for the hint calculation logic
        try:
            model_map = self.bot.get_probability_map()
            # TODO : Use dependency injection to implement the logic to only show the map if it has been changed
            # if self.old_probability_map is None or (model_map != self.old_probability_map):
            #     self.old_probability_map = model_map
            #     probability_map = model_map
            # else:
            #     probability_map = None
        except AttributeError:
            probability_map = None
        grid, grid_view = game.visible_grid()

        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = self.hint_buttons[row][col]

                if grid_view[row, col]:
                    self.update_visible_button(button, int(grid[row, col]))
                else:
                    self.update_hint_button(
                        button,
                        hints[row, col],
                        None if probability_map is None else probability_map[row, col],
                    )

    def update_visible_button(self, button: tk.Button, value: int):
        text_button = str(value) if value != 0 else ""
        color = "lightgrey"
        button.config(text=text_button, bg=color)

    def get_color(self, value: float):
        """
        Takes a value between -1 and +1 and returns a corresponding color
        that is a gradient between orange (-1) and cyan (+1).

        :param value: A float between -1 and 1
        :return: A hex color string for Tkinter (e.g., '#RRGGBB')
        """
        # Ensure value is between -1 and 1
        value = max(-1, min(1, value))

        # Normalize value to be between 0 and 1 (0 -> orange, 1 -> cyan)
        normalized_value = (value + 1) / 2

        # Compute the red, green, and blue components based on the normalized value
        red = int(
            (1 - normalized_value) * 255 + normalized_value * 0
        )  # Orange has 255 red, Cyan has 0 red
        green = int(
            (1 - normalized_value) * 165 + normalized_value * 255
        )  # Orange has 165 green, Cyan has 255 green
        blue = int(
            (1 - normalized_value) * 0 + normalized_value * 255
        )  # Orange has 0 blue, Cyan has 255 blue

        # Return the color as a hex string
        return f"#{red:02x}{green:02x}{blue:02x}"

    def update_hint_button(
        self, button: tk.Button, hint_value: Any, probability: float | None = None
    ):
        if hint_value == 1:
            text_button = "V"  # Indicate safe move
            color = "green"
        elif hint_value == -1:
            text_button = "X"  # Indicate dangerous move (possible mine)
            color = "red"
        else:
            if probability is not None:
                text_button = "P"
                color = self.get_color(2 * probability - 1)
            else:
                text_button = ""
                color = "gray"

        button.config(text=text_button, bg=color)

    def calculate_best_moves(self, game: Game):
        # Placeholder logic for best move calculation
        hints = np.zeros_like(game.player_grid_view, dtype=np.int64)

        bot_action = self.bot.action(*game.visible_grid())
        hints[*bot_action] = 1

        try:
            known_mines = self.bot.get_known_mines()
            hints[known_mines] = -1
        except AttributeError:
            pass

        # 0 -> player_grid_view
        # 1 -> Safe
        # -1 -> Mine
        return hints
