import numpy as np

"""
Define players for the game of TicTacToe.
All players received canonical states, meaning they should only behave as the first player.

"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        valids = self.game.getValidMoves(state, 1)
        available_actions = []
        for i in range(len(valids)):
            if valids[i]:
                available_actions.append(i)
        return np.random.choice(available_actions)


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, state):
        valid = self.game.getValidMoves(state, 1)
        action = -1
        while action not in valid or not valid[action] or valid.count(1) != 0:
            print("Please input your move, the format is \"row col\". Example: 1 1 meaning the [1, 1] position")
            move = input()
            x, y = [int(x) for x in move.split(' ')]
            action = self.game.n * x + y
            if valid[action]:
                return action
            else:
                print("Invalid move")

        return action
