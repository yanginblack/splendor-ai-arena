import numpy as np

"""
Define players for the game of Splendor.
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
        valids = self.game.getValidMoves(state, 1)
        # if only one valid action and it is the pass action, return pass
        if np.sum(valids) == 1 and valids[47]:
            return 47
        print("Valid actions: ", valids)
        print("Please input your action:")
        while True:
            action = int(input())
            if valids[action]:
                break
            else:
                print("Invalid action, please try again.")
        return action