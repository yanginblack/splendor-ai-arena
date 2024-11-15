import numpy as np
import pickle 
import torch
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

class GreedyPlayer():
    def __init__(self, game):
        self.game = game
        # create a weight for each action
        self.weights = np.ones(self.game.getActionSize())
        # purchaseing a card is the most important action
        for i in range(12, 15):
            self.weights[i] = 48 # 48 is the heighest weight, purchasing card from reservation slots
        for i in range(11, -1, -1):
            self.weights[i] = 47 - i # the lower the number, the higher the weight, purchasing card from market
        
        # then reserve a card for highest level
        for i in range(26, 14, -1):
            self.weights[i] = 35 - i

    def play(self, state):
        valids = self.game.getValidMoves(state, 1)
        priorities = self.weights * valids
        return np.argmax(priorities)
    
class SPLPlayer():
    def __init__(self, game):
        self.game = game
        #self.model = pickle.load(open("model_data_updated_data10_new60.pkl", 'rb'))
        self.model = pickle.load(open("model_data1.pkl", 'rb'))

    def play(self, state):
        val, indices = torch.sort(self.model.predictaction(torch.tensor(state).float()), descending = True)
        i = 0
        valids = self.game.getValidMoves(state, 1)
        indices = indices.tolist()
        action = indices[i]
        while not valids[action]:
            i+=1
            action = indices[i]
        return action