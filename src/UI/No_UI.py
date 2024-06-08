from src.Game import Game
from src.Players.Player_Interface import Player_Interface


class No_UI:
    def start(self, game: Game, player:Player_Interface):
        while not game.is_ended(): # Player has not discovered all cases
            next_action = player.action(*game.visible_grid())
            game.action(*next_action)
        result = game.result()
        return result