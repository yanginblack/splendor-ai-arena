# from TicTacToe.TicTacToeGame import TicTacToeGame as Game
# from TicTacToe.TicTacToePlayers import *
from Splendor.SplendorGame import SplendorGame as Game
from Splendor.SplendorPlayers import *

class GameConfig():
    def __init__(self):
        self.stopThreshold = 100
        self.game = Game()
    def initialize_game(self):
        return self.game
    # Given the name of player function, return the initiate player
    
    def initialize_player(self, name):
        # return the player function with given name
        return globals()[name](self.game).play
    
    def get_display(self):
        return Game.display
