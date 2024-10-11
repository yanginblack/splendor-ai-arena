import logging
from termcolor import colored
from tqdm import tqdm

log = logging.getLogger(__name__)


class PlayGround():
    """
    This class is used to host game play between given players.
    """

    def __init__(self, game, stopThreshold):
        """
        Input:
            players: players list that contains player functions
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.players = []
        self.game = game
        self.stopThreshold = stopThreshold

    def playGame(self, player_ids, display_flag=False):
        """
        Executes one episode of a game.

        Returns:
            winner: player who won the game (1 if player1, 2 if player2)
        """
        curPlayer = 1
        state = self.game.getInitState()
        it = 0

        if display_flag:
            print("Initial board:")
            self.game.display(state)
        # A flag to terminate the game when the depth is reached
        terminated = False
        result = self.game.getGameEnded(state, curPlayer)

        while result[0] == 0:
            action = self.players[curPlayer-1](self.game.getCanonicalForm(state, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), 1)
            action = self.game.translateCanonicalAction(action, curPlayer)
            if action < 42: # only count actions that's not discarding or holding
                it += 1
            # check if the action is valid
            if not valids[action]:
                log.error(f'Action {action} is not valid!')
                log.error(f'valids = {valids}')
                assert valids[action]

            if display_flag:
                print("Turn ", str(it), "Player ", str(player_ids[curPlayer-1]), "Action: ", self.game.displayAction(action))
            state, curPlayer = self.game.getNextState(state, curPlayer, action)
            if display_flag:
                self.game.display(state)
            
            # if the game is terminated by exceeding the stopThreshold, break the loop
            if it == self.stopThreshold:
                terminated = True
                break

            result = self.game.getGameEnded(state, curPlayer)
        
        if display_flag:
            print("Game over: Turn ", str(it), "Result ", str(result))
            self.game.display(state)

        return result if not terminated else [-1]

    def playGames(self, num, players, player_ids, display_flag=False):
        """
        play the number of games

        Returns:
            The list of each player performance: [(winning, losing, draws), (winning, losing, draws), ...]
            winning: the list of the player winning times
            losing: the list of the player losing times
            draws: the list of the player draws times also including the games exceeding the stopThreshold
        """
        self.players = players
        player_performance = [{"winning": 0, "losing": 0, "draws": 0} for _ in range(len(players))]

        for _ in tqdm(range(num), desc="PlayGround Starting"):
            gameResult = self.playGame(player_ids, display_flag=display_flag)
            for i in range(len(players)):
                if i+1 in gameResult:
                    player_performance[i]["winning"] += 1
                elif -1 in gameResult:
                    player_performance[i]["draws"] += 1
                else:
                    player_performance[i]["losing"] += 1

        return player_performance