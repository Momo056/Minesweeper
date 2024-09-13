from tkinter import Tk

from src.Players.Minesweeper_bot import Minesweeper_bot
from src.UI.GUI_Hint_Grid import GUI_Hint_Grid
from src.Game import Game
from src.Grid import Grid
from src.UI.GUI_User_Inputs import GUI_User_Inputs
from src.App.App import App

if __name__ == '__main__':
    # Parameters
    grid_size = (10, 10)
    mine_percent = 0.15
        
    # Main code
    master = Tk()
    grid = Grid(*grid_size, mine_percent)
    game = Game(grid)
    minesweeper_bot = Minesweeper_bot(True, True)
    gui = GUI_Hint_Grid(minesweeper_bot, master)
    gui.start(game)
