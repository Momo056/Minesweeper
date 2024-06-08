from src.Game import Game
from src.Players.Player_Interface import Player_Interface
from src.UI.GUI_User_Inputs import GUI_User_Inputs
import tkinter as tk
import numpy as np


class GUI_Bot_Inputs(GUI_User_Inputs):
    def __init__(self, master: tk.Tk | None = None, delay:int=1000):
        super().__init__(master)
        self.master.bind("<FocusIn>", self.on_focus_in)
        self.delay = delay

    def on_focus_in(self, event):
        self.bot_play()

    def bot_play(self):
        if not self.game.is_ended():
            next_action = self.player.action(*self.game.visible_grid())
            self.on_button_click(self.game, *next_action)

            # Add flags
            try:
                flags = np.argwhere(self.player.known_mines)
                for x, y in flags:
                    if not self.flags[x, y]:
                        self.on_right_click(self.game, x, y)
            except:
                pass
            self.master.after(self.delay, self.bot_play)



    def start(self, game: Game, player: Player_Interface):
        self.game = game
        self.player = player
        return super().start(game)