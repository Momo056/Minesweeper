from math import prod
import tkinter as tk
from typing import Any, Callable
import numpy as np
from src.Players.Player_Interface import Player_Interface
from src.Game import Game


class GUI_Hint_Map:
    def __init__(
        self,
        bot: Player_Interface,
        map_provider: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
        master: tk.Tk | None = None,
    ):
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
        self.initialize_all_grid(game)

        # Launch the app
        self.master.mainloop()

        return game.result()

    def create_widgets(self, game: Game):
        current_row = 0

        # Centering configuration: allow expansion on left and right of the main frame
        self.master.grid_columnconfigure(0, weight=1)

        # Create a frame to contain all three grids side by side
        self.main_frame = tk.Frame(self.master)
        self.main_frame.grid(row=current_row, column=0, padx=10, pady=10)

        # Add each grid
        self.hint_grid_frame, self.hint_grid_label = self._create_grid(self.main_frame, 'Bot action', 0)
        self.grid_frame, self.player_grid_label = self._create_grid(self.main_frame, 'Player grid', 1)
        self.probability_grid_frame, self.probability_grid_label = self._create_grid(self.main_frame, 'Mine probability prediction', 2)
        
        self.player_grid_label.configure(font='Helvetica 18 bold')

        current_row += 1
        # Add space for end of game feedback
        self.feedback_label = tk.Label(self.master, text='')
        self.feedback_label.grid(row=current_row, column=0, pady=10)

        current_row += 1
        # Add space for end of game feedback
        self.status_bar = tk.Label(self.master, text='')
        self.status_bar.grid(row=current_row, column=0, pady=10)

        current_row += 1
        # Add new button for playing generated action
        self.generated_action_button = tk.Button(
            self.master,
            text="Play bot action",
            font='Helvetica 16',
            command=lambda: self.play_generated_action(game),
        )
        self.generated_action_button.grid(row=current_row, column=0, pady=10)

        current_row += 1
        self.option_frame = tk.Frame(self.master)
        self.option_frame.grid(row=current_row, column=0, padx=10, pady=10)
        self._create_option(self.option_frame)

        # Maximize the window on startup
        self.master.state('zoomed')
        self.master.focus_force()

    def _create_option(self, outer_frame: tk.Frame):
        self.hint_checkbox = self._abstract_checkbox(outer_frame, 0, 'Bot action', self.show_hint_grid, self.hide_hint_grid)
        self.probability_checkbox = self._abstract_checkbox(outer_frame, 1, 'Probability prediction', self.show_probability_grid, self.hide_probability_grid)

    def _abstract_checkbox(self, outer_frame: tk.Frame, col: int, text: str, on_func: Callable, off_func: Callable):
        # Probability
        var = tk.BooleanVar()
        def update_probability_checkbox():
            if var.get():
                on_func()
            else:
                off_func()
        checkbox = tk.Checkbutton(outer_frame, variable=var, onvalue=True, offvalue=False, text=text, command=update_probability_checkbox)
        checkbox.grid(row=0, column=col, padx=10)
        checkbox.select()

        return checkbox
    
    def show_hint_grid(self):
        self._show_abstract_grid(self.hint_grid_frame, self.hint_grid_label, 0)
        self.generated_action_button.config(state=tk.NORMAL)

    def hide_hint_grid(self):
        self._hide_abstract_grid(self.hint_grid_frame, self.hint_grid_label)
        self.generated_action_button.config(state=tk.DISABLED)

    def show_probability_grid(self):
        self._show_abstract_grid(self.probability_grid_frame, self.probability_grid_label, col=2)

    def hide_probability_grid(self):
        self._hide_abstract_grid(self.probability_grid_frame, self.probability_grid_label)

    def _show_abstract_grid(self, frame:tk.Frame, label: tk.Label, col: int):
        frame.grid(row=1, column=col, padx=10)
        label.grid(row=0, column=col, padx=10)

    def _hide_abstract_grid(self, frame:tk.Frame, label: tk.Label):
        frame.grid_forget()
        label.grid_forget()

    
    def _create_grid(self, outer_frame: tk.Frame, text: str, col: int):
        # Add label above hint grid
        label = tk.Label(outer_frame, text=text, font='Helvetica 15')
        label.grid(row=0, column=col, padx=10, pady=5)
        
        # Create hint grid frame (middle)
        frame = tk.Frame(outer_frame)
        
        self._show_abstract_grid(frame, label, col)

        return frame, label


    def play_generated_action(self, game: Game):
        if self.generated_action is not None:
            try:
                flags = self.bot.get_known_mines()
                for row, col in np.argwhere(flags):
                    self.flags[row, col] = True
            except Exception as e:
                print(e)
            # Play the action and update the grid
            self.on_button_click(game, *self.generated_action)
        else:
            print("No action is generated yet.")

    def initialize_all_grid(self, game: Game):
        # Initialize flag storage
        self.flags = np.zeros_like(game.player_grid_view)

        self.initialize_grid(game)
        self.initialize_hint_grid(game)
        self.initialize_probability_grid(game)

        self.update_all_grids(game)

    def initialize_grid(self, game: Game):
        def button_updater(button: tk.Button, row: int, col: int):
            button.bind(
                "<Button-3>",
                lambda e, r=row, c=col: self.on_right_click(game, row, col),
            )
            button.config(command=lambda r=row, c=col: self.on_button_click(game, row, col))

        self.buttons = self._initialize_abstract_grid(game, self.grid_frame, button_updater)

    def initialize_hint_grid(self, game: Game):
        self.hint_buttons = self._initialize_abstract_grid(game, self.hint_grid_frame, lambda *_:None)
        
    def initialize_probability_grid(self, game: Game):
        self.probability_buttons = self._initialize_abstract_grid(game, self.probability_grid_frame, lambda *_:None)

    def _initialize_abstract_grid(self, game: Game, frame: tk.Frame, button_updater: Callable[[tk.Button, int, int], None]):
        # Create the buttons/grid for the probabilities
        button_matrix: list[list[tk.Button]] = []
        for row in range(game.grid.grid_shape()[0]):
            row_buttons = []
            for col in range(game.grid.grid_shape()[1]):
                button = tk.Button(frame, width=2, height=1)
                button.grid(row=row, column=col)
                row_buttons.append(button)

                button_updater(button, row, col) # Extern modification
            button_matrix.append(row_buttons)

        return button_matrix

    def on_button_click(self, game: Game, row: int, col: int):
        game.action(row, col)
        self.update_all_grids(game)

    def on_right_click(self, game: Game, row: int, col: int):
        self.flags[row, col] = not self.flags[row, col]
        self.update_all_grids(game)
        
    def update_status_bar(self, game: Game):
        n_flag = int(np.sum(np.logical_and(self.flags, ~game.player_grid_view)))
        n_covered = int(np.sum(1-game.player_grid_view))
        n_unknown = n_covered - n_flag
        text=f"{game.grid.n_bomb - n_flag} mines left\n{n_flag} flags\n{n_unknown} unknown boxes"
        
        self.status_bar.config(text=text)

            

    def update_feedback(self, game: Game):
        if game.is_ended():
            if game.result():
                text = 'YOU WIN !'
                fg = 'green'
            else:
                text = 'YOU LOSE'
                fg = 'red'
        else:
            text = ''
            fg = 'black'
        # Initialize feedback
        self.feedback_label.config(text=text, fg=fg, font='Helvetica 30 bold')

    def update_all_grids(self, game: Game):
        self.update_grid(game)
        self.update_hint_grid(game)
        self.update_probability_grid(game)
        self.update_status_bar(game)
        self.update_feedback(game)

    def update_grid(self, game: Game):
        if game.is_ended():
            if game.result():
                # Win the game -> Show all mines as flags
                def button_updater(button: tk.Button, row: int, col: int):
                    if game.grid.mines[row, col]:
                        text_button = "F"
                        color = "yellow"
                    else:
                        text_button = ""
                        color = "gray"

                    button.config(text=text_button, bg=color)
            else:
                # Lose the game -> Show all mines
                def button_updater(button: tk.Button, row: int, col: int):
                    if game.grid.mines[row, col]:
                        text_button = "M"
                        color = "red"
                    else:
                        text_button = ""
                        color = "gray"

                    button.config(text=text_button, bg=color)
        else:
            def button_updater(button: tk.Button, row: int, col: int):
                if self.flags[row, col]:
                    text_button = "F"
                    color = "yellow"
                else:
                    text_button = ""
                    color = "gray"

                button.config(text=text_button, bg=color)

        self._update_abstract_grid(game, self.buttons, button_updater)

    def update_hint_grid(self, game: Game):
        if game.is_ended():
            return
        hints = self.calculate_best_moves(game)

        button_updater = lambda b, r, c : self.update_hint_button(b, hints[r, c])

        self._update_abstract_grid(game, self.hint_buttons, button_updater)

    def update_probability_grid(self, game: Game):
        if self.map_provider is None:
            probability_map = None
        else:
            probability_map = self.map_provider(*game.visible_grid())

        if probability_map is None:
            button_updater = lambda b, r, c : self.update_probability_button(b, None)
        else:
            button_updater = lambda b, r, c : self.update_probability_button(b, probability_map[r, c])

        self._update_abstract_grid(game, self.probability_buttons, button_updater)


    def _update_abstract_grid(self, game: Game, button_matrix: list[list[tk.Button]], button_updater: Callable[[tk.Button, int, int], None]):
        for row in range(game.grid.grid_shape()[0]):
            for col in range(game.grid.grid_shape()[1]):
                button = button_matrix[row][col]

                if game.player_grid_view[row, col]:
                    self.update_visible_button(button, int(game.grid.grid[row, col]), is_mine=game.grid.mines[row, col])
                else:
                    # Injected code
                    button_updater(button, row, col)

    def update_visible_button(self, button: tk.Button, value: int, is_mine:bool = False):
        if is_mine:
            text_button = "M"
            color = "red"
        else:
            text_button = str(value) if value != 0 else ""
            color = "lightgrey"
        button.config(text=text_button, bg=color)

    def update_hint_button(self, button: tk.Button, hint_value: Any):
        if hint_value == 1:
            text_button = "V"  # Safe move
            color = "green"
        elif hint_value == -1:
            text_button = "X"  # Mine
            color = "red"
        else:
            text_button = ""
            color = "gray"
        button.config(text=text_button, bg=color)

    def update_probability_button(self, button: tk.Button, probability: float | None):
        if probability is not None:
            two_digit = round((1 - probability) * 100)
            # Format the probability to two decimal places and ensure it starts with a dot (e.g., ".39" for 0.38856)
            text_button = (
                f".{two_digit}" if two_digit < 100 else "1"
            )  # Slice off the leading '0' to keep the format '.XX'
            color = self.get_color(probability)
        else:
            text_button = ""
            color = "gray"
        button.config(text=text_button, bg=color)

    def get_color(self, value: float):
        value = max(0, min(1, value))
        normalized_value = value

        red = int((1 - normalized_value) * 255)
        green = int((1 - normalized_value) * 165 + normalized_value * 255)
        blue = int(normalized_value * 255)

        return f"#{red:02x}{green:02x}{blue:02x}"

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
