import logging
from termcolor import colored
from tqdm import tqdm

log = logging.getLogger(__name__)


class PlayGround():
    """
    This class is used to host game play among given players.
    """

    def __init__(self, game, players, player_names, stopThreshold):
        """
        Input:
            players: player list that contains playable function
            game: Game object
            stopThreshold: the maximum number of actions for a game
        Provides:
            playGame: a function to play one game
            playGames: a function to play multiple games
        """
        self.players = players
        self.game = game
        self.player_names = player_names
        self.stopThreshold = stopThreshold

    def playGame(self, players, player_names, display_flag=False):
        """
        Executes one episode of a game.

        Returns:
            winner: player who won the game (1 if player1, 2 if player2), -1 if draw
        """
        curPlayer = 1
        state = self.game.getInitState()
        it = 0
        cur_players = players

        if display_flag:
            print("Initial board:")
            self.game.display(state)
        # A flag to terminate the game when the depth is reached
        terminated = False
        result = self.game.getGameEnded(state, curPlayer)

        while result[0] == 0:
            action = cur_players[curPlayer-1](self.game.getCanonicalForm(state, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), 1)
            action = self.game.translateCanonicalAction(action, curPlayer)
            # Splendor specific, should be removed in the future
            if action < 42: # only count actions that's not discarding or holding
                it += 1
            # check if the action is valid
            if not valids[action]:
                log.error(f'Action {action} is not valid!')
                log.error(f'valids = {valids}')
                assert valids[action]

            if display_flag:
                print("Turn ", str(it), "Player ", str(curPlayer), player_names[curPlayer-1], "Action: ", self.game.displayAction(action))
            state, curPlayer, reward = self.game.getNextState(state, curPlayer, action)
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

    def playGames(self, num, display_flag=False, rotate_flag=False):
        """
        play the number of games

        Returns:
            The list of each player performance: [(winning, losing, draws), (winning, losing, draws), ...]
            winning: the list of the player winning times
            losing: the list of the player losing times
            draws: the list of the player draws times also including the games exceeding the stopThreshold
        """
        player_performance = [{"winning": 0, "losing": 0, "draws": 0} for _ in range(len(self.players))]
        player_ids = [i+1 for i in range(len(self.players))]
        player_names = self.player_names.copy()
        rotated_game_plays = [num // len(self.players) for _ in range(len(self.players))] if rotate_flag else [num]
        cur_players = self.players.copy()

        for played_times in rotated_game_plays:
            for _ in tqdm(range(played_times), desc="PlayGround Starting for {} games".format(played_times)):
                gameResult = self.playGame(cur_players, player_names, display_flag=display_flag)
                for i in range(len(player_ids)):
                    if i+1 in gameResult:
                        player_performance[player_ids[i]-1]["winning"] += 1
                    elif -1 in gameResult:
                        player_performance[player_ids[i]-1]["draws"] += 1
                    else:
                        player_performance[player_ids[i]-1]["losing"] += 1
            # Rotate the players
            cur_players = cur_players[1:] + cur_players[:1]
            player_names = player_names[1:] + player_names[:1]
            player_ids = player_ids[1:] + player_ids[:1]
        return player_performance