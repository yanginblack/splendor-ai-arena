class BaseGame():
    """
    This class specifies the base Game template. 
    To define your own game, subclass this class and implement the functions below.
    Players count support multiple players, player 1 is the first player, then player 2, etc.
    
    """
    def __init__(self):
        pass

    def getInitState(self):
        """
        The initial state of the game, state should be markov state containing all the necessary information for players to make optimal decision.
        Please be aware that the state should be independent of player and your game class. We don't recommend to hold information in the game class 
        since the state passed in might be the canonical version with player 1's perspective. 
        Returns:
            startState: a representation of the initialized state (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getStateSize(self):
        """
        Returns:
            (x,y,z): a tuple of state dimensions. Recommended three dimensions as maximum. If it's only two dimensions, use (1, x, y).
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, state, player, action):
        """
        This function is used to get the next state of the game given the current state, the player and the action taken by the player.
        
        Input:
            state: current state
            player: current player (1, 2, ..., n)
            action: action taken by current player

        Returns:
            nextState: state after applying action
            nextPlayer: player who plays in the next turn. Please be aware that next player may not be the player on sequence.
            e.g. in a two-player game, player 1 may play twice before player 2 plays once.
        """
        pass

    def getValidMoves(self, state, player):
        """
        Input:
            state: current state
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current state and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, state):
        """
        Input:
            state: current state

        Returns:
            r: 0 if game has not ended. -1 for draw, 1 for player 1 win, 2 for player 2 win, etc.
               
        """
        pass

    def getCanonicalState(self, state, player):
        """
        This function is used to train model with same view as player 1 for all players. 
        Input:
            board: current board
            player: current player (1, 2, ..., n)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be the player1's view of the given player. 
                            For example, in the game of Go, the canonical form of black is the same as the board itself, 
                            while the canonical state of white is the board with all stones inverted, and view from the other side of board.
        """
        pass
    
    def translateCanonicalAction(self, action, player):
        """
        This function is used to translate the given canonical action to the action in the original state space.
        Input:
            action: action in the canonical state space
            player: current player (1, 2, ..., n)

        Returns:
            action: action in the original state space
        """
        pass

    def getSymmetries(self, state, pi):
        """
        This function is used to generate symmetrical forms of the state for training.
        Some games could rotate or flip boards with no difference but with same best action, 
        we can treat them as same state.
        Input:
            state: current state
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(state,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def hashRepresentation(self, state):
        """
        Input:
            state: current state

        Returns:
            a hash representation of the state used as a key for caching and training.
        """
        pass

