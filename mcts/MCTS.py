import numpy as np
import math

EPS = 1e-8

class MCTS:
    """
    Monte Carlo Tree Search
    game: game class
    nnet: neural network model
    args: arguments {
        cpuct: exploration constant
        num_simulations: number of simulations to run
        player_count: number of players
    }
    """
    def __init__(self, game, pnet, vnet, args):
        self.game = game
        self.pnet = pnet
        self.vnet = vnet
        self.args = args
        
        self.Qsa = {} # Q(s, a), Q value of action a taken on state s, key is (state_hash, action)
        self.Nsa = {} # N(s, a), how many times action a has been taken on state s, key is (state_hash, action)
        self.Ns = {} # N(s), how many times state s has been visited, key is (state_hash)
        self.Ps = {} # P(s, a), prior probability of action a taken on state s, key is (state_hash, action)
        
        self.Vs = {} # valid moves of s, key is (state_hash)
    
    def play(self, canonical_state, temperature=0):
        """
        Search the best action with the given state using given neural network.
        If temperature is 0, choose the most visited action, which is pure exploitation.
        Otherwise, temperature is 1, then use a decayed temperature to explore the action space.
        Return the policy pi: a list of action probabilities.
        """

        for _ in range(self.args['num_simulations']):
            self.run(canonical_state, 1, depth=0)
        
        state_hash = self.game.hashRepresentation(canonical_state)
        visit_counts = np.array([self.Nsa[(state_hash, action)] if (state_hash, action) in self.Nsa else 0 for action in range(self.game.getActionSize())])
        valid_moves = self.game.getValidMoves(canonical_state, 1)
        visit_counts = visit_counts * valid_moves
        total_visits = float(visit_counts.sum())
        if total_visits == 0:
            # happens when the end state is reached
            valid_moves = valid_moves / sum(valid_moves)
            return valid_moves

        # if temperature == 0, choose the most visited action
        if temperature < 0.1:
            best_action = np.argmax(visit_counts)
            pi = np.zeros(self.game.getActionSize())
            pi[best_action] = 1
            return pi

        else:
            # if temperature == 1, choose the action with a probability proportional
            # return based on action visited probability, also give more chance to the less visited actions
            # visit_counts = visit_counts ** (1 / temperature) hide this for now
            counts = [x ** (1. / temperature) for x in visit_counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            return probs

    def run(self, state, player_id, depth):
        """
        Run the MCTS algorithm. The function runs for one iteration of MCTS.
        From the root node to the leaf node and backpropagate the result.
        When meets a leaf node (a node that first time visited), get probs and value from neural network. 
            And expend the node with the action probabilities.
        When meets the end of game, backpropagate the result.
        Otherwise, use UCB to select the next action. 
        Please note from the same state, even the same action may lead to different states.
        Return: value, winning player ids.
        """
        if depth > self.args['maxDepth']:
            # learn nothing
            return 0, []
        canonical_state = self.game.getCanonicalForm(state, player_id)
        state_hash = self.game.hashRepresentation(canonical_state)
        
        # if the game is ended, backpropagate the result.
        if self.game.getGameEnded(state, player_id)[0] != 0:
            return 1, self.game.getGameEnded(state, player_id)
        
        if state_hash not in self.Ps:
            # This is the leaf node.
            # neural network predicts from canonical state
            # action_probs, value = self.nnet.predict(canonical_state)
            value = self.vnet.predict(canonical_state)
            action_probs = self.pnet.predict(canonical_state)[0]
            # action_probs = [1] * self.game.getActionSize()

            valid_moves = self.game.getValidMoves(state, player_id)
            # issue of action_probs being all 0s expect 1, try to make probs more evenly distributed
            action_probs = [max(prob, self.args['ubc_epsilon']) for prob in action_probs]
            action_probs = action_probs * valid_moves

            action_probs /= sum(action_probs)

            self.Ps[state_hash] = action_probs
            self.Ns[state_hash] = 0
            self.Vs[state_hash] = valid_moves
            # the returned value represents estimated V of current state for the player who is about to play
            return value, [player_id] # setting player_id to this value (might be negative)

        # this node is not first time visited.
        self.Ns[state_hash] += 1 # add one visit
        valid_moves = self.Vs[state_hash] # get valid moves
        action_probs = self.Ps[state_hash] # get action probabilities

        # select the child node with the highest UCB score.
        best_score = float('-inf')
        best_action = -1
        for action, prob in enumerate(action_probs):
            if prob > 0:
                score = self.ucb_score(state_hash, action, prob)
                if score > best_score:
                    best_score = score
                    best_action = action
        
        # get the next state of the current state
        next_state, next_player_id, _ = self.game.getNextState(state, player_id, best_action)
        # recursively run the MCTS algorithm
        value, winning_player_ids = self.run(next_state, next_player_id, depth+1)
        # update the result
        # value is for the player id in the list of winning_player_ids
        v = value if player_id in winning_player_ids else -value
        
        if (state_hash, best_action) not in self.Qsa:
            self.Qsa[(state_hash, best_action)] = v
            self.Nsa[(state_hash, best_action)] = 1
        else:
            self.Qsa[(state_hash, best_action)] = (self.Nsa[(state_hash, best_action)] * self.Qsa[(state_hash, best_action)] + v) / (self.Nsa[(state_hash, best_action)] + 1)
            self.Nsa[(state_hash, best_action)] += 1

        return value, winning_player_ids
    
    def ucb_score(self, state_hash, action, prob):
        """
        Calculate the UCB score for a child node.
        """
        if (state_hash, action) in self.Nsa:
            return self.Qsa[(state_hash, action)] + \
                self.args['cpuct'] * prob * \
                math.sqrt(self.Ns[state_hash]) / (
                1 + self.Nsa[(state_hash, action)])
        else:
            return self.args['cpuct'] * prob * math.sqrt(self.Ns[state_hash] + EPS)

