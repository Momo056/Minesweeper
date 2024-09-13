from tkinter import Tk

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
    gui = GUI_User_Inputs(master)
    gui.start(game)
