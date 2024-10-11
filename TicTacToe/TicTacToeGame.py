import sys
sys.path.append('..')
from BaseGame import BaseGame
import numpy as np

"""
Game class implementation for the game of TicTacToe.
This is an example of how to implement a game class

"""
class TicTacToeGame(BaseGame):
    """
    In this game, board is a 2D nXn matrix, 0 is empty, 1 is player 1, 2 is player 2
    """
    def __init__(self, n=3):
        self.n = n
        self.players = [1, 2]
        self.board = [None for _ in range(n)]
        for i in range(n):
            self.board[i] = [0 for _ in range(n)]

    def getInitState(self):
        return np.array(self.board)

    def getStateSize(self):
        return (1, self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, state, player, action):
        row = action // self.n
        col = action % self.n
        state[row][col] = player
        next_player = self.players[player%2]
        return (state, next_player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] == 0:
                    valids[i*self.n+j] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        empty_count = 0
        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] != 0:
                    if i+1 < self.n and i+2 < self.n and board[i+1][j] == board[i][j] and board[i+2][j] == board[i][j]:
                        return board[i][j]
                    elif j+1 < self.n and j+2 < self.n and board[i][j+1] == board[i][j] and board[i][j+2] == board[i][j]:
                        return board[i][j]
                    elif i+1 < self.n and i+2 < self.n and j+1 < self.n and j+2 < self.n and board[i+1][j+1] == board[i][j] and board[i+2][j+2] == board[i][j]:
                        return board[i][j]
                    elif i-1 >= 0 and i-2 >= 0 and j+1 < self.n and j+2 < self.n and board[i-1][j+1] == board[i][j] and board[i-2][j+2] == board[i][j]:
                        return board[i][j]
                else:
                    empty_count += 1
        if empty_count == 0:
            return -1
        return 0

    def translateCanonicalAction(self, action, player):
        return action

    def getCanonicalForm(self, board, player):
        if player == 2:
            for i in range(self.n):
                for j in range(self.n):
                    if board[i][j] == 1:
                        board[i][j] = 2
                    elif board[i][j] == 2:
                        board[i][j] = 1
        return board

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == 2: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
