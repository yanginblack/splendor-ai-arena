from PlayGround import PlayGround
import numpy as np
from Splendor.SplendorGame import SplendorGame as Game
from Splendor.SplendorPlayers import *
import torch

class SPL(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout):
        super(SPL, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, bias = True)
        
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim, bias = True)

        # Probability of an element getting zeroed
        self.dropout = torch.nn.Dropout(p = dropout)
        self.activation = torch.nn.ReLU()


    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, state):
        out = self.layer1(state)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layer2(out)

        return out

    def predict(self, state_prime):
      out = self.layer1(state_prime)
      out = self.activation(out)
      out = self.layer2(out)

      return torch.max(out, dim = 1).values

    def predictaction(self, state_prime):
      out = self.layer1(state_prime)
      out = self.activation(out)
      out = self.layer2(out)

      return out
    

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
#player_names = ["SPLPlayer", "GreedyPlayer", "RandomPlayer"]
player_names = ["SPLPlayer", "RandomPlayer", "RandomPlayer"]
#player_names = ["GreedyPlayer", "SPLPlayer", "RandomPlayer"]
#player_names = ["RandomPlayer", "RandomPlayer", "RandomPlayer"]
num_games = 300
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