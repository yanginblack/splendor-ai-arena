from PlayGround import PlayGround
import numpy as np
from Splendor.SplendorGame import SplendorGame as Game
from Splendor.SplendorPlayers import *

def rotate_player_order(players):
    players = players[1:] + [players[0]]
    return players

def displayResults(player_performance, player_ids):
    for i in range(len(player_performance)):
        print(f"Player {player_ids[i]} performance: {player_performance[i]}")
"""
use this script to play any number of agents against each other.
"""
num_players = 2
player_names = ["RandomPlayer", "RandomPlayer"]
num_games = 1
rotate_flag = False
display_flag = True
max_steps = 100

players = []
player_ids = []



print("Starting the game...")
game = Game(num_players)
playground = PlayGround(game, max_steps)
for i in range(num_players):
    players.append(globals()[player_names[i]](game).play)
    player_ids.append(i+1)

if rotate_flag:
    num_games_per_player = num_games // num_players
    for i in range(num_players):
        player_performance = playground.playGames(num_games_per_player, players, player_ids, display_flag)
        displayResults(player_performance, player_ids)

        players = rotate_player_order(players)
        player_ids = rotate_player_order(player_ids)
else:
    player_performance = playground.playGames(num_games, players, player_ids, display_flag)
    displayResults(player_performance, player_ids)




