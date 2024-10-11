from PlayGround import PlayGround
import numpy as np
from GameConfig import GameConfig

config = GameConfig()

def rotate_player_order(players):
    players = players[1:] + [players[0]]
    return players

def displayResults(player_performance, player_ids):
    for i in range(len(player_performance)):
        print(f"Player {player_ids[i]} performance: {player_performance[i]}")
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
num_players = 3

player_names = ["HumanPlayer", "RandomPlayer", "RandomPlayer"]

num_games = 1

rotate_flag = False

display_flag = True

players = []
player_ids = []
for i in range(num_players):
    players.append(config.initialize_player(player_names[i]))
    player_ids.append(i+1)


print("Starting the game...")
game = config.initialize_game()
playground = PlayGround(game, config.stopThreshold)

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




