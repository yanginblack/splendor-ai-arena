from PlayGround import PlayGround
import numpy as np
from Splendor.SplendorGame import SplendorGame as Game
from Splendor.SplendorPlayers import *

def rotate_player_order(players):
    players = players[1:] + [players[0]]
    return players

def displayResults(player_performance, player_names):
    for i in range(len(player_performance)):
        print(f"Player {i+1} {player_names[i]} performance: {player_performance[i]}")
"""
use this script to play any number of agents against each other.
"""
num_players = 3
player_names = ["RandomPlayer", "RandomPlayer", "RandomPlayer"]
num_games = 30
rotate_flag = True
display_flag = False
max_steps = 500

players = []

print("Starting the game...")
game = Game(num_players)
for i in range(num_players):
    players.append(globals()[player_names[i]](game).play)

playground = PlayGround(game, players, player_names, max_steps)

player_performance = playground.playGames(num_games, display_flag, rotate_flag)
displayResults(player_performance, player_names)