import sys
import os
sys.path.append('..')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from BaseGame import BaseGame
import numpy as np
from constants import CARDS, NOBLES
import copy
from termcolor import colored
import random

"""
Game class implementation for the game of Splendor.

"""
class SplendorGame(BaseGame):
    """
    State(len = 300): 
        (0 - 131) 12 cards read to be purchased, 4 cards for level1, 4 cards for level2, 4 cards for level3. 
        Each card has 11 digits: first 5 are the required gems of the card, in the order of white, red, green, blue and brown.
        The next 5 represent the number of gems this card provides. e.g. Card provide a red gem, it should be [0 1 0 0 0].
        The last digit is the card's points.
        e.g. [0 1 0 2 2 0 0 0 0 0 1 0] is the card required 1 red gem, 2 blue gems, 2 brown gems. Provides 1 brown gem. No points.
        (132 - 155) 4 nobles, 6 digits for 1 noble. e.g. [1 1 1 1 1 3] is a noble that requires 1 of each gem, provides 3 points.
        (156 - 161) public remaining gems: white, red, green, blue, brown and gold.
        (162 - 207) player1 info 46 digits
            162 - 167: player 1 gems: white, red, green, blue, brown and gold.
            168 - 172: player 1 permanent gems: white, red, green, blue, and brown.
            173: player 1 points
            174 - 206: player 1 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
            207: player 1 acquired cards.
        208 - 253: player 2 info
            208 - 213: player 2 gems: white, red, green, blue, brown and gold.
            214 - 218: player 2 permanent gems: white, red, green, blue, and brown.
            219: player 2 points
            220 - 252: player 2 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
            253: player 2 acquired cards.
        254 - 299: player 3 info
            254 - 259: player 3 gems: white, red, green, blue, brown and gold.
            260 - 264: player 3 permanent gems: white, red, green, blue, and brown.
            265: player 3 points
            266 - 298: player 3 reserved cards. 11 digits for 1 card. Maximum reserved 3 cards.
            299: player 3 acquired cards.
    Action (len = 48):    
        0 - 14: 12+3 cards purchase, 4 cards per row, row 0 is level 1, row 1 is level 2, row 2 is level 3. (0-11), 3 reserved card purchase(12-14), 
        15 - 26: 12 cards reserve, rows can columns are the same as cards purchase, (15-26), 5 taking two gems of (white, red, green, blue and brown) (27-31), 
        27 - 41: C53 to take 3 in different color(32-36), 5 for discarding gem (white, red, green, blue and brown) (37-41), 1 for pass (42)
        42 - 47: for discarding gems when the any player has more than 10 gems in their round, all other players only have 1 action: pass, and that player
        only have five discarding actions until this player has 10 gems. 
    """
    def __init__(self, player_number = 3):
        self.state_size = 301
        self.action_size = 48
        self.player_points = [173, 219, 265]
        self.player_reserved_cards = [[174, 185, 196], [220, 231, 242], [266, 277, 288]]
        self.player_permanent_gems = [168, 214, 260]
        self.player_gems = [162, 208, 254]
        self.player_states = [[162, 208], [208, 254], [254, 300]] # [start, end) of player states
        self.public_remaining_gems = 156
        self.public_nobels = [132, 138, 144, 150]
        self.take_gems = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        self.level_cards = [0, 44, 88]
        self.player_acquired_cards = [207, 253, 299]
        self.gem_colors = ["white", "red", "green", "blue", "dark_grey", "yellow"]
        self.state = np.zeros(self.state_size)
        self.nobles = np.zeros(20)
        self.player_number = player_number
        self.all_cards = [
            CARDS[0:40],
            CARDS[40:70],
            CARDS[70:90]
        ]

    def getInitState(self):
        # set up initial state: random 4 cards for level 1, 2, 3. 
        for i in range(4):
            level1_card = random.choice(self.all_cards[0])
            self.state[i*11+self.level_cards[0]:(i+1)*11+self.level_cards[0]] = level1_card
            self.all_cards[0].remove(level1_card)

            level2_card = random.choice(self.all_cards[1])
            self.state[i*11+self.level_cards[1]:(i+1)*11+self.level_cards[1]] = level2_card
            self.all_cards[1].remove(level2_card)

            level3_card = random.choice(self.all_cards[2])
            self.state[i*11+self.level_cards[2]:(i+1)*11+self.level_cards[2]] = level3_card
            self.all_cards[2].remove(level3_card)

        # pick 4 random nobels.
        self.nobles = random.sample(NOBLES, 4)
        for i in range(4):
            self.state[self.public_nobels[i]:self.public_nobels[i]+6] = self.nobles[i]

        # five gems available for each color as a three player game.
        self.state[self.public_remaining_gems:self.public_remaining_gems+6] = [5, 5, 5, 5, 5, 5]
        
        return np.array(self.state)

    def getStateSize(self):
        return (1, 1, self.state_size)

    def getActionSize(self):
        # return number of actions
        return self.action_size

    # Pass in state, player and action. Execute the action and return the next state and next player.
    # Returns:
    #    nextState: state after applying action
    #    nextPlayer: player who plays in the next turn. Please be aware that next player may not be the player on sequence.
    def getNextState(self, state, player_id, action):
        nextState = state
        player = player_id - 1
        # purchase one of public cards
        if action < 12:
            # purchase one of public cards
            nextState = self._purchaseCard(state, player, state[action*11:(action+1)*11])
            # replace the card with the next card in the deck. Use all zeros if deck is empty.
            new_card = random.choice(self.all_cards[action//4]) if len(self.all_cards[action//4]) > 0 else np.zeros(11)
            nextState[action*11:(action+1)*11] = new_card
            self.all_cards[action//4].remove(new_card)
        # purchase one of reserved cards
        elif action < 15:
            nextState = self._purchaseCard(state, player, state[self.player_reserved_cards[player][action-12]:self.player_reserved_cards[player][action-12]+11])
            # set the reserved card slot to be empty
            nextState[self.player_reserved_cards[player][action-12]:self.player_reserved_cards[player][action-12]+11] = np.zeros(11)
        # reserve one of public cards
        elif action < 27:
            # find one empty slot for the player
            reserve_card_index = 0
            for i in range(3):
                if sum(state[self.player_reserved_cards[player][i]:self.player_reserved_cards[player][i]+11]) == 0:
                    reserve_card_index = i
                    break
            # reserve one of the public cards
            nextState[self.player_reserved_cards[player][reserve_card_index]:self.player_reserved_cards[player][reserve_card_index]+11] = nextState[(action-15)*11:(action-15+1)*11]
            
            # add one golden gem to the player, if there is any.
            if nextState[self.public_remaining_gems+5] > 0:
                nextState[self.player_gems[player]+5] += 1
                nextState[self.public_remaining_gems+5] -= 1
            # replace the card with the next card in the deck. Use all zeros if deck is empty.
            new_card = random.choice(self.all_cards[(action-15)//4]) if len(self.all_cards[(action-15)//4]) > 0 else np.zeros(11)
            nextState[(action-15)*11:(action-15+1)*11] = new_card
            self.all_cards[(action-15)//4].remove(new_card)
        # take gems
        elif action < 42:
            for i in self.take_gems[action-27]:
                if nextState[self.public_remaining_gems + i] > 0:   
                    nextState[self.public_remaining_gems + i] -= 1
                    nextState[self.player_gems[player] + i] += 1
        # discard gems
        elif action < 47:
            nextState[self.public_remaining_gems + action-42] += 1
            nextState[self.player_gems[player] + action-42] -= 1
        return nextState, player_id%self.player_number+1


    def getValidMoves(self, state, player_id):
        valid_moves = np.zeros(self.action_size)
        player = player_id - 1
        # check if any player has more than 10 gems:
        for i in range(3):
            if sum(state[self.player_gems[i]:self.player_gems[i]+6]) > 10:
                if i == player:
                    for j in range(5):
                        if state[self.player_gems[i]+j] > 0:
                            valid_moves[42+j] = 1 # discard gem when the player has any.
                else:
                    valid_moves[47] = 1 # other players have to pass.
                return valid_moves

        for i in range(self.action_size):
            # purchase one of public cards
            if i < 12:
                if self._isValidPurchase(state[self.player_states[player][0]:self.player_states[player][1]], state[i*11:(i+1)*11]):
                    valid_moves[i] = 1
            # purchase one of reserved cards
            elif i < 15:
                if self._isValidPurchase(state[self.player_states[player][0]:self.player_states[player][1]], state[self.player_reserved_cards[player][i-12]:self.player_reserved_cards[player][i-12]+11]):
                    valid_moves[i] = 1
            # reserve one of public cards
            elif i < 27:
                # valid only if this player has empty sports
                for j in range(3):
                    if sum(state[self.player_reserved_cards[player][j]:self.player_reserved_cards[player][j]+11]) == 0:
                        valid_moves[i] = 1
                        break
            # taking 2 gems of the same color is invalid unless that color has more than 4 public gems before taking.
            elif i < 32:
                if state[self.public_remaining_gems+i-27] >= 4:
                    valid_moves[i] = 1
            # take 3 gems are always valid unless any player has more than 10 gems.(player can take 3, 2, 1, 0 gems as they want)
            elif i < 42:
                valid_moves[i] = 1
            # discard gems are always invalid if no player has more than 10 gems.
        return valid_moves


    # [0] if not ended, [1] if player 1 won, [2] if player 2 won, [3] for player 3 won.
    # multiple winners (draw) return the list of winners.
    def getGameEnded(self, state, player_id):
        # game will last until player3 complete action.
        if player_id != self.player_number: 
            return [0]
        # get player points
        player_points = [state[self.player_points[0]], state[self.player_points[1]], state[self.player_points[2]]]

        max_points = max(player_points)
        # if no one reach 15 points, the game is not ended.
        if max_points < 15:
            return [0]
        
        # if only one player reach 15 points, that player won.
        if player_points.count(max_points) == 1:
            return [player_points.index(max_points) + 1]
        # if more than one player reach 15 points, check their development cards, the one with less cards win. If still tie then players share victory.
        cards_count = [state[self.player_acquired_cards[0]], state[self.player_acquired_cards[1]], state[self.player_acquired_cards[2]]]
        for i in range(3):
            if player_points[i] != max_points:
                cards_count[i] = float('inf') # only count winning player's cards.

        min_cards = min(cards_count)
        winners = []
        for i in range(3):
            if cards_count[i] == min_cards:
                winners.append(i+1)
        return winners
        
    # translate canonical action (the action viewed as player 1's perspective) to the actual action according to the actual player.
    # In this game, the canonical action is always the same as the actual action
    def translateCanonicalAction(self, action, player_id):
        return action

    # translate the board to the canonical form (the form viewed as player 1's perspective). 
    # Don't do it in place, create a new array.
    # In this game, just copy the state[self.player_states[player][0]:self.player_states[player][1]] to the state[0:self.player_states[0][1]-self.player_states[0][0]]
    def getCanonicalForm(self, state, player_id):
        newState = copy.deepcopy(state)
        if player_id != 1:
            temp = state[self.player_states[0][0]:self.player_states[0][1]]
            newState[self.player_states[0][0]:self.player_states[0][1]] = newState[self.player_states[player_id-1][0]:self.player_states[player_id-1][1]]
            newState[self.player_states[player_id-1][0]:self.player_states[player_id-1][1]] = temp
        return newState

    def stringRepresentation(self, state):
        return state.tostring()

    # check whether player has enough gems: permanent gems + current gems >= required gems from card.
    # golden gems can be used to replace any color of required gems
    def _isValidPurchase(self, player_state, card):
        # check if the card slot is empty: all cards have one provided gem. 
        if sum(card[5:10]) == 0:
            return False
        # check if the card can be purchased with the player's current gems and permanent gems.
        required_golden_gems = 0
        for i in range(5):
            required_golden_gems += max(0, card[i] - player_state[i] - player_state[i+6])
        if required_golden_gems > player_state[5]:
            return False
        return True

    # purchase of one card:
    # 1. Reduce player's current gems by the card's required gems.
    # 2. Return the paid gems to the public gem pool.
    # 3. Increase player's permanent gem by the card's provided gem.
    # 4. Increase player's points by the card's points.
    # 5. Increase player's acquired cards by 1.
    # 6. Check whether the player satisfied the noble requirement and get points if so, then remove the noble from public pool.
    def _purchaseCard(self, state, player, card):
        # when doing purchase, try to use as less golden gems as possible.
        paid_golden_gems = 0
        for i in range(5):
            required_gems = max(0, card[i] - state[self.player_permanent_gems[player]+i])
            paid_golden_gems += max(0, required_gems - state[self.player_gems[player]+i])
            paid_gems = min(required_gems, state[self.player_gems[player]+i])
            state[self.player_gems[player]+i] -= paid_gems # reduce player's current gems
            state[self.public_remaining_gems+i] += paid_gems # return paid gems to public gem pool
        state[self.player_gems[player]+5] -= paid_golden_gems # reduce player's golden gems
        state[self.public_remaining_gems+5] += paid_golden_gems # add paid golden gems to public gem pool

        # increase player's permanent gems
        for i in range(5):
            state[self.player_permanent_gems[player]+i] += card[i+5]
        # increase player's points
        state[self.player_points[player]] += card[10]
        # increase player's acquired cards
        state[self.player_acquired_cards[player]] += 1
        # check whether the player satisfied the noble requirement: here is a rule modification, a player can take multiple nobles at once, the original rule required only one noble per action.
        acquired_nobels = []
        for i in range(4):
            if sum(state[self.public_nobels[i]:self.public_nobels[i]+6]) > 0: # noble is not taken
                valid_noble = True
                for j in range(5):
                    if state[self.public_nobels[i]+j] > state[self.player_permanent_gems[player]+j]:
                        valid_noble = False
                        break
                if valid_noble:
                    state[self.player_points[player]] += state[self.public_nobels[i]+5]
                    state[self.public_nobels[i]:self.public_nobels[i]+6] = np.zeros(6)
        return state

    # display a more readable action name instead of a pure number.
    def displayAction(self, action):
        if action < 12:
            return "Purchase Level " + str(action//4+1) + " Card " + str(action%4+1)
        elif action < 15:
            return "Purchase Reserved Card " + str(action-12+1)
        elif action < 27:
            return "Reserve Level " + str((action-15)//4+1) + " Card " + str((action-15)%4+1)
        elif action < 42:
            color_string = ""
            for i in self.take_gems[action-27]:
                color_string += colored("\u25CF", self.gem_colors[i]) + " "
            return "Take " + color_string
        elif action < 47:
            return "Discard " + colored("\u25CF", self.gem_colors[action-42]) + " "
        else:
            return "Hold"

    def display(self, state):    
        # print public info section
        print("="*58, end="")
        print("  public  ", end="")
        print("="*58)

        # First line showing public gems
        print("Available Gems: ", end="")
        for i in range(6):
            print(self._displayGem(state[self.public_remaining_gems+i], i, 5), "   ", end="")
        print()

        # Second line showing nobles
        print("Nobles: ", end="")
        for i in range(4):
            common_gems = 3
            # if noble slot is empty, print 15 spaces
            if sum(state[self.public_nobels[i]:self.public_nobels[i]+6]) == 0:
                print(" "*15, end="")
                continue
            
            noble_string = "["
            for j in range(5):
                if state[self.public_nobels[i]+j] > 0:
                    noble_string += colored("\u25AE", self.gem_colors[j])*int(state[self.public_nobels[i]+j]) + " "
                    if state[self.public_nobels[i]+j] > 3:
                        common_gems = 4
                        noble_string += "  "
            # remove two extra space if noble_string has length larger than 15
            if common_gems == 4:
                noble_string = noble_string[:-2]
            
            noble_string += str(common_gems) + "]"
            print(noble_string, " "*5, end="")
        print()

        # Showing public cards
        cards_list = [[], [], []]
        for i in range(12):
            cards_list[i//4].append(self._displayCard(state[i*11:i*11+11]))
        for level in range(3):
            for i in range(8):
                if i == 3:
                    print("Level " + str(level+1) + ": ", end="")
                else:
                    print(" "*9, end="")
                for card in cards_list[level]:
                    print(card[i], end="")
                    print(" "*6, end="")
                print()
        
        # player info section    
        print("="*58, end="")
        print("  players  ", end="")
        print("="*58)

        # first line showing player points and cards acquired
        print("Player 1 Points:", self._displayNumber(state[self.player_points[0]]), "Cards: ", self._displayNumber(state[self.player_acquired_cards[0]]), end="") # 28
        print(" "*20, end="")
        print("Player 2 Points:", self._displayNumber(state[self.player_points[1]]), "Cards: ", self._displayNumber(state[self.player_acquired_cards[1]]), end="") # 28
        print(" "*20, end="")
        print("Player 3 Points:", self._displayNumber(state[self.player_points[2]]), "Cards: ", self._displayNumber(state[self.player_acquired_cards[2]])) # 28

        # second line showing player gems
        for player in range(3):
            print("Gems: ", end="")
            displayed_gems = 0
            for i in range(6):
                if state[self.player_gems[player]+i] > 0:
                    displayed_gems += 1
                    print(self._displayGem(state[self.player_gems[player]+i], i, 5), " ", end="")
            while displayed_gems < 6:
                print(" "*7, end="")
                displayed_gems += 1
        print()

        # third line showing player tokens (acquired permanent gems)
        for player in range(3):
            print("P Germs: ", end="")
            displayed_gems = 0
            for i in range(5):
                if state[self.player_permanent_gems[player]+i] > 0:
                    displayed_gems += 1
                    print(self._displayGem(state[self.player_permanent_gems[player]+i], i, 5, True), " ", end="")
            while displayed_gems < 5:
                print(" "*7, end="")
                displayed_gems += 1
            print(" "*4, end="")
        print()
        # fourth section showing player reserved cards
        cards_list = []
        for player in range(3):
            for i in range(3):
                cards_list.append(self._displayCard(state[self.player_reserved_cards[player][i]:self.player_reserved_cards[player][i]+11]))
        
        for i in range(8):
            for card_index, card in enumerate(cards_list):
                print(" ", end="")
                print(card[i], end="")
                print(" "*3, end="")
                if (card_index+1)%3 == 0:
                    print(" "*11, end="")
            print()
        print("="*126)
        print()
        print()
        print()    

    def _displayGem(self, gem_count, gem_id, required_space, isPermanent=False):
        if isPermanent:
            gem_str = "\u29BE"
        else:
            gem_str = "\u25CF"
        display_string = ""
        if gem_count == 0:
            display_string += " "*required_space
        elif gem_count <=required_space:
            display_string += " "*int(required_space - gem_count)
            display_string += colored(gem_str, self.gem_colors[gem_id])*int(gem_count)
        else:
            display_string += " "*int(required_space-2)
            display_string += str(int(gem_count))
            display_string += colored(gem_str, self.gem_colors[gem_id])
        return display_string

    def _displayNumber(self, number):
        return str(int(number))

    def _displayCard(self, card):
        display_card = []
        display_required_gems = []

        # return 8*6 empty cards if card is empty
        if sum(card[5:10]) == 0:
            for i in range(8):
                if i == 0 or i == 7:
                    display_card.append("-"*8)
                else:
                    display_card.append("|" + " "*6 + "|")
            return display_card

        for i in range(5):
            if card[i] > 0:
                display_required_gems.append(self._displayGem(card[i], i, 5))

        for i in range(8):
            if i == 0 or i == 7:
                display_card.append("-"*8)
            elif i == 1:
                color_index = list(card[5:10]).index(1)
                display_card.append("|" + str(int(card[10])) + " "*4 + colored("\u25CF", self.gem_colors[color_index]) + "|")
            elif i >= 7 - len(display_required_gems):
                display_card.append("| " + display_required_gems[i-(7-len(display_required_gems))] + "|")
            else:
                display_card.append("|" + " "*6 + "|")
        
        return display_card