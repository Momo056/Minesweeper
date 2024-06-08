from src.Game import Game
from src.Players.Player_Interface import Player_Interface


class Command_Line_UI:
    def start(self, game: Game, player:Player_Interface):
        print('Game start')
        print(f'Grid size : {game.grid.grid_shape()}')
        while not game.is_ended(): # Player has not discovered all cases
            next_action = player.action(*game.visible_grid())
            print(f'Player plays {next_action}')
            
            game.action(*next_action)
        
        result = game.result()
        if result:
            print('Player win')
        else:
            print('Player lose')
        return result