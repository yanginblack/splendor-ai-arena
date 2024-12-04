import sys
sys.path.append('..')

import logging
from MCTS import MCTS
import numpy as np
import os
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from tqdm import tqdm
from PlayGround import PlayGround
from Splendor.SplendorPlayers import RandomPlayer
from Splendor.SplendorPlayers import GreedyPlayer

log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play learning.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        # load the previous model if there is any
        if self.isFileExists(self.args['temp_file']):
            self.nnet.load_checkpoint(self.args['checkpoint'], self.args['temp_file'])

        # load the training examples from the previous iteration
        self.trainExamplesHistory = [] # the examples from latest iteration
        if self.isFileExists(self.args['temp_file'] + ".examples"):
            filename = os.path.join(self.args['checkpoint'], self.args['temp_file'] + ".examples")
            with open(filename, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()

    def learn(self):
        """
        Performs {self.args.iterations} iterations with {self.args.episodes} episodes of self-play
        in each iteration. 
        At the end of each iteration, coach trains the neural network with all gathered examples.
        Then use the latest nnet play against previous model, accept new model 
        only if it wins >= {self.args.updateThreshold} fraction of games.
        """
        for i in range(1, self.args['numIters'] + 1):
            log.info(f'Starting Iter #{i} ...')
            trainExamples = deque([], maxlen=self.args['iterQueueMaxLen'])
            
            temperature = 1 if i < int(self.args['numIters']/2) else 0
            log.info(f"temperature: {temperature}")

            for _ in tqdm(range(self.args['episodes']), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                trainExamples += self.runEpisode(temperature)
            # save the iteration examples to the history 
            self.trainExamplesHistory.append(trainExamples)

            if len(self.trainExamplesHistory) > self.args['trainingDataMaxLen']:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                # TODO: think of the golden data for training
                self.trainExamplesHistory.pop(0)
            if i%self.args['evaluation_interval'] == 0:
                self.saveTrainingExamples()

            if i%self.args['evaluation_interval'] == 0:
                # shuffle examples before training
                trainExamples = []
                for e in self.trainExamplesHistory:
                    trainExamples.extend(e)
                shuffle(trainExamples)

                # keep a copy in the temp folder
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.args['temp_file'])

                self.nnet.train(trainExamples)
            # test the new model
            if i%self.args['evaluation_interval'] == 0:
                nmcts = MCTS(self.game, self.nnet, self.args)

                log.info('PLAYING AGAINST PREVIOUS VERSION')
                players = [lambda x: np.argmax(nmcts.play(x, 0)),
                            RandomPlayer(self.game).play,
                                RandomPlayer(self.game).play]
                player_names = ["NEW_MCTS", "RandomPlayer", "RandomPlayer"]

                playground = PlayGround(self.game, players, player_names, self.args['maxSteps'])
                performance = playground.playGames(self.args['arenaCompare'], display_flag=False, rotate_flag=True)
                for player_name, player_performance in zip(player_names, performance):
                    log.info(f"{player_name} wins: {player_performance['winning']}, loses: {player_performance['losing']}, draws: {player_performance['draws']}")
                
                if performance[0]['winning'] + performance[0]['losing'] != 0 and float(performance[0]['winning']) / (performance[0]['winning'] + performance[0]['losing'] + performance[0]['draws']) >= self.args['updateThreshold']:
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename="random_defeat.pth.tar")
                
                players = [lambda x: np.argmax(nmcts.play(x, 0)),
                            GreedyPlayer(self.game).play,
                                GreedyPlayer(self.game).play]
                player_names = ["NEW_MCTS", "GreedyPlayer", "GreedyPlayer"]

                playground = PlayGround(self.game, players, player_names, self.args['maxSteps'])
                performance = playground.playGames(self.args['arenaCompare'], display_flag=False, rotate_flag=True)
                for player_name, player_performance in zip(player_names, performance):
                    log.info(f"{player_name} wins: {player_performance['winning']}, loses: {player_performance['losing']}, draws: {player_performance['draws']}")
                if performance[0]['winning'] + performance[0]['losing'] != 0 and float(performance[0]['winning']) / (performance[0]['winning'] + performance[0]['losing'] + performance[0]['draws']) >= self.args['updateThreshold']:
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename="greedy_defeat.pth.tar")
                
                players = [lambda x: np.argmax(nmcts.play(x, 0)),
                            RandomPlayer(self.game).play,
                                GreedyPlayer(self.game).play]
                player_names = ["NEW_MCTS", "RandomPlayer", "GreedyPlayer"]

                playground = PlayGround(self.game, players, player_names, self.args['maxSteps'])
                performance = playground.playGames(self.args['arenaCompare'], display_flag=False, rotate_flag=True)
                for player_name, player_performance in zip(player_names, performance):
                    log.info(f"{player_name} wins: {player_performance['winning']}, loses: {player_performance['losing']}, draws: {player_performance['draws']}")

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (performance[0]['winning'], performance[0]['losing'], performance[0]['draws']))
                if performance[0]['winning'] + performance[0]['losing'] == 0 or float(performance[0]['winning']) / (performance[0]['winning'] + performance[0]['losing'] + performance[0]['draws']) < self.args['updateThreshold']:
                    log.info('REJECTING NEW MODEL')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.args['best_file'])
                    break

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainingExamples(self):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.args['temp_file'] + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def isFileExists(self, filename):
        folder = self.args['checkpoint']
        return os.path.exists(os.path.join(folder, filename))

    def runEpisode(self, temperature):
        """
        This function executes one episode of self-play
        As the game is played, each turn is added as a training example to
        trainData. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainData.

        Returns:
            trainData: a list of examples of the form (canonicalBoard, currPlayer, pi, v)
                           pi is the MCTS informed policy vector, v is value of current state
        """
        trainData = []
        state = self.game.getInitState()
        self.curPlayer = 1
        episodeStep = 0
        
        while True:
            episodeStep += 1

            # temperature adjustment due to last episode length. Assuming each run has similar length
            # temperature = 1 if episodeStep < self.preEpisodeSteps//2 else 0
            canonicalState = self.game.getCanonicalForm(state, self.curPlayer)
            if self.curPlayer == 1:
                pi = self.mcts.play(canonicalState, temperature=temperature)
                best_action = np.random.choice(self.game.getActionSize(), p=pi)
            elif self.curPlayer == 2:
                best_action = GreedyPlayer(self.game).play(canonicalState)
                pi = [0] * self.game.getActionSize()
                pi[best_action] = 1
            else:
                best_action = RandomPlayer(self.game).play(canonicalState)
                pi = [0] * self.game.getActionSize()
                pi[best_action] = 1
            trainData.append([canonicalState, self.curPlayer, pi])

            state, self.curPlayer, _ = self.game.getNextState(state, self.curPlayer, best_action)
            r = self.game.getGameEnded(state, self.curPlayer)
            if r[0] != 0 or episodeStep == self.args['maxSteps']:
                self.preEpisodeSteps = episodeStep
                # if it's current player, set r, if it's not, set -r
                # if r is 2, always set reward to -1, otherwise set reward to r
                if episodeStep == self.args['maxSteps']:
                    result = [] # when exceeding max steps, make sure all players get -1 reward
                else:
                    result = r
                return [(x[0], x[2], 1 * ((-1) ** (x[1] not in result))) for x in trainData]
