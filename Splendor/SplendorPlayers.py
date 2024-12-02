import numpy as np
from mcts.MCTS import MCTS
# from mcts.NNet import NNetWrapper as nn

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
        print("Valid actions:")
        for i in range(len(valids)):
            if valids[i]:
                print("action ", i, ": ", self.game.displayAction(i), end="\t")
        print()
        print("Please input your action as a number:")
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
        # purchase reserve card if possible
        for i in range(12, 15):
            if valids[i]:
                return i
        # purchase the most expensive card
        for i in range(11, -1, -1):
            if valids[i]:
                return i
        # reserve card if possible
        for i in range(15, 27):
            if valids[i]:
                return i
        # then grab gems
        possible_actions = []
        for i in range(27, 42):
            if valids[i]:
                possible_actions.append(i)
        if len(possible_actions) == 0:
            for i in range(len(valids)):
                if valids[i]:
                    possible_actions.append(i)
        
        return np.random.choice(possible_actions)

class AdvancedGreedyPlayer():
    def __init__(self, game):
        self.game = game
        self.EPS = 0.1
        self.safety_cost = 3
        self.player = 0
        self.greedy_player = GreedyPlayer(self.game)
        self.player_gems = self.game.player_gems
        self.take_gems = self.game.take_gems
        self.player_reserved_cards = self.game.player_reserved_cards
        self.player_permanent_gems = self.game.player_permanent_gems
        self.public_nobels = self.game.public_nobels
        self.public_remaining_gems = self.game.public_remaining_gems

    
    def play(self, state):
        valids = self.game.getValidMoves(state, 1)
        if valids[47]:
            return 47
        # get cards list
        cards_list = []
        target_card = None
        for i in range(15):
            if i < 12:
                card = state[i*11:(i+1)*11]
            else:
                card = state[self.player_reserved_cards[self.player][i-12]:self.player_reserved_cards[self.player][i-12]+11]
            
            cards_list.append((self.get_card_cost(state, card), self.get_card_value(state, card), i, card))

        # sort cards_list based on the price-performance ratio
        cards_list.sort(key=lambda x: x[1]/(x[0]+self.EPS), reverse=True)
        # find the most available card with the highest price-performance ratio
        for card in cards_list:
            if card[0] <= self.safety_cost and self.card_is_available(state, card):
                target_card = card
                break
        # print("target card: ", target_card)
        # print("cards list: ", cards_list)
        # if player has to discard gem, discard gems not affecting the target card.
        if sum(valids[42:47]) == 5:
            return self.discard_gems(state, target_card)

        # if player can purchase the target card, purchase it.
        if valids[target_card[2]]:
            return target_card[2]

        # card is not reserved and other players can buy this card, then reserve it.
        if target_card[2] < 12 and self.should_reserve(state, target_card):
            return target_card[2] + 15
        
        # otherwise, grab gems to purchase this card.
        return self.grab_gems(state, target_card)
    
    def discard_gems(self, state, target_card):
        original_cost = target_card[0]
        required_gem_list = state[target_card[2]*11+0:target_card[2]*11+5] - state[self.player_permanent_gems[self.player]+0:self.player_permanent_gems[self.player]+5] - state[self.player_gems[self.player]+0:self.player_gems[self.player]+5]
        for i in range(5):
            if required_gem_list[i] < 0:
                # print("discard gem: ", i)
                return 42+i
        # print("random discard")
        return np.random.choice(range(42, 47))

    def card_is_available(self, state, card):
        required_gem_list = card[3][0:5] - state[self.player_permanent_gems[self.player]+0:self.player_permanent_gems[self.player]+5] - state[self.player_gems[self.player]+0:self.player_gems[self.player]+5]
        required_gem_list = [max(0, x) for x in required_gem_list]
        for i in range(5):
            if state[self.public_remaining_gems+i] - required_gem_list[i] < 0:
                # print("this card is not available", card)
                return False
        # print("this card is available", card)
        return True
    
    def should_reserve(self, state, card):
        # check if current player has empty slots.
        if sum(state[self.player_reserved_cards[self.player][0]:self.player_reserved_cards[self.player][0]+11]) == 0 \
            or sum(state[self.player_reserved_cards[self.player][1]:self.player_reserved_cards[self.player][1]+11]) == 0 \
            or sum(state[self.player_reserved_cards[self.player][2]:self.player_reserved_cards[self.player][2]+11]) == 0:
            # print("no empty slot")
            return False
        
        # check if other players can buy this card.
        for i in range(1, 3):
            other_player_valids = self.game.getValidMoves(state, i+1)
            if other_player_valids[card[2]]:
                # print("other player can buy this card", card, "player", i+1)
                return True
        # print("no other player can buy this card", card)
        return False
    
    def grab_gems(self, state, card):
        required_gem_list = card[3][0:5] - state[self.player_permanent_gems[self.player]+0:self.player_permanent_gems[self.player]+5] - state[self.player_gems[self.player]+0:self.player_gems[self.player]+5]
        required_gem_list = [max(0, x) for x in required_gem_list]
        valids = self.game.getValidMoves(state, 1)
        achieved_gems = -float('inf')
        best_action = -1
        for i in range(27, 42):
            temp_gem_list = required_gem_list.copy()
            grabbed_gems = 0
            if valids[i]:
                for gem in self.take_gems[i-27]:
                    if temp_gem_list[gem] > 0:
                        temp_gem_list[gem] -= 1
                        grabbed_gems += 1
                if grabbed_gems > achieved_gems:
                    achieved_gems = grabbed_gems
                    best_action = i
        # print("decided to grab gem: ", self.game.displayAction(best_action))
        return best_action if best_action != -1 else self.greedy_player.play(state)
    
    def get_card_value(self, state, card):
        value = card[10]
        for j in range(len(self.public_nobels)):
            adjusted_noble_list = state[self.public_nobels[j]:self.public_nobels[j]+5] - state[self.player_permanent_gems[self.player]+0:self.player_permanent_gems[self.player]+5]
            adjusted_noble_list = [max(0, x) for x in adjusted_noble_list]
            for color_id in range(5):
                if card[5+color_id] == 1 and adjusted_noble_list[color_id] > 0:
                    value += state[self.public_nobels[j]+5]/sum(adjusted_noble_list)
        return value

    def get_card_cost(self, state, card):
        required_gem_list = card[0:5] - state[self.player_permanent_gems[self.player]+0:self.player_permanent_gems[self.player]+5] - state[self.player_gems[self.player]+0:self.player_gems[self.player]+5]
        required_gem_list = [max(0, x) for x in required_gem_list]
        return max(0, sum(required_gem_list)-state[self.player_gems[self.player]+5])

class DQNPlayer():
    def __init__(self, game, agent):
        self.game = game
        self.agent = agent

    def play(self, state):
        q_values = self.agent.predict(state)[0]
        valids = self.game.getValidMoves(state, 1)
        action = np.argmax(q_values*valids)

        if valids[action] == 0:
            random_actions = [i for i in range(len(valids)) if valids[i]]
            action = np.random.choice(random_actions)
        return action

class SIMDQNPlayer():
    def __init__(self, game, agent):
        self.game = game
        self.agent = agent

    def play(self, state):
        simplified_state = self.game.getSimplifiedState(state, 1)
        q_values = self.agent.predict(simplified_state)[0]
        valids = self.game.getValidMoves(state, 1)
        action = np.argmax(q_values*valids)

        if valids[action] == 0:
            
            random_actions = [i for i in range(len(valids)) if valids[i]]
            action = np.random.choice(random_actions)
        return action

class MCTSPlayer():
    def __init__(self, game, pnet, vnet, args):
        self.game = game
        self.mcts = MCTS(game, pnet, vnet, args)

    def play(self, state):
        pi = self.mcts.play(state, 0)
        return np.argmax(pi)

    
class SPLPlayer():
    def __init__(self, game):
        self.game = game
        self.model = pickle.load(open("model_data_updated_data10_new60.pkl", 'rb')) # MLP2
        # self.model = pickle.load(open("model_data1.pkl", 'rb')) # MLP1

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
