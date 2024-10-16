from PlayGround import PlayGround
import numpy as np
from Splendor.SplendorGame import SplendorGame as Game
from Splendor.SplendorPlayers import *
import logging
import coloredlogs
from dqn.model import modelWrapper as dqn_net
from mcts.MCTS import MCTS
from mcts.VNet import NNetWrapper as mcts_vnet
from mcts.PNet import NNetWrapper as mcts_pnet
import torch
import sys
sys.path.append('./dqn')
sys.path.append('./mcts')
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

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

player_names = ["MCTSPlayer", "GreedyPlayer", "GreedyPlayer"]
num_games = 30
rotate_flag = True
display_flag = False
max_steps = 500 # for one single player, the maximum number of steps in one game

players = []


print("Starting the game...")
game = Game(num_players)

print("Initializing players...")
dqn_args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
}
dqn_agent = dqn_net(game, dqn_args)
dqn_agent.loadModel(folder='./dqn/', filename='best.pth.tar')

mcts_args = {
    'cpuct': 2,
    'num_simulations': 80,
    'maxDepth': 100,
    'ubc_epsilon': 0.02,
    'cuda': torch.cuda.is_available(),
}
mcts_pnet_model = mcts_pnet(game)
mcts_pnet_model.loadModel(folder='./mcts/', filename='policy.pth.tar')
mcts_vnet_model = mcts_vnet(game)
mcts_vnet_model.load_checkpoint(folder='./mcts/', filename='value.pth.tar')


for i in range(num_players):
    if player_names[i] == "MCTSPlayer":
        players.append(MCTSPlayer(game, mcts_pnet_model, mcts_vnet_model, mcts_args).play)
    elif player_names[i] == "DQNPlayer":
        players.append(DQNPlayer(game, dqn_agent).play)
    else:
        players.append(globals()[player_names[i]](game).play)

playground = PlayGround(game, players, player_names, max_steps)

player_performance = playground.playGames(num_games, display_flag, rotate_flag)
displayResults(player_performance, player_names)