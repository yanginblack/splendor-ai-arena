import sys
sys.path.append('..')

import logging
import numpy as np
from tqdm import tqdm
from PlayGround import PlayGround
from Splendor.SplendorPlayers import DQNPlayer
from Splendor.SplendorPlayers import SIMDQNPlayer
from Splendor.SplendorPlayers import RandomPlayer
from Splendor.SplendorPlayers import GreedyPlayer
from Splendor.SplendorPlayers import AdvancedGreedyPlayer
import os
from pickle import Pickler, Unpickler

log = logging.getLogger(__name__)

class Train():
    """
    This class is used for training the neural network.
    It runs maximum iterations in the args['numIters'], and evaluate the agent performance after each evaluateInterval iterations.
    If the agent can win more than args['updateThreshold'] of the games against the previous best model, the new model will be accepted. Then training ends.
    """
    def __init__(self, game, qnet, args):
        self.game = game
        # initialize Q network and target network
        self.qnet = qnet
        self.args = args
        self.targetNet = self.qnet.__class__(self.game, args)
        # load the model
        if self.checkFileExists('best.model.tar'):
            self.qnet.loadModel(folder=self.args['checkpoint'], filename='best.model.tar')
            self.targetNet.loadModel(folder=self.args['checkpoint'], filename='best.model.tar')
        else:
            self.qnet.saveModel(folder=self.args['checkpoint'], filename='temp.tar')
            self.targetNet.loadModel(folder=self.args['checkpoint'], filename='temp.tar')
        self.epsilon = args['startEpsilon']
        
        self.selfPlay = SelfPlay(game, args)
        # read training history
        if self.checkFileExists('trainingData.examples'):
            with open(os.path.join(self.args['checkpoint'], 'trainingData.examples'), 'rb') as f:
                self.trainingHistory = Unpickler(f).load()
        else:
            self.trainingHistory = []
        log.info(f"Loaded {len(self.trainingHistory)} training examples")


    def start(self):
        for i in range(1, self.args['numIters']+1):
            log.info(f'Starting Iter #{i} ...')
            # calculate the episilon
            self.epsilon = max(
                self.args['minEpsilon'], 
                self.args['startEpsilon'] - ((self.args['startEpsilon'] - self.args['minEpsilon']) / self.args['numIters']) * i
            )
            print(f'epsilon: {self.epsilon}')
            # self-play numEpisodes games and get training data
            trainingData = self.selfPlay.play(self.qnet, self.targetNet, self.epsilon, self.args['numEpisodes'])
            trainingDataToStore = trainingData.copy()
            print(f"trainingData length: {len(trainingData)}")

            # extend the training
            requiredMoreTraining = self.args['trainingDataMaxLen'] - len(trainingData) - len(self.trainingHistory)
            if requiredMoreTraining > 0:
                log.warning(f"Not enough training data to meet the required length. Required: {requiredMoreTraining}, Available: {len(trainingData)}")
                trainingData.extend(self.trainingHistory)
            else:
                sample_ids = np.random.choice(len(self.trainingHistory), -requiredMoreTraining, replace=False)
                trainingData.extend([self.trainingHistory[i] for i in sample_ids])
            log.info(f"Train with {len(trainingData)} examples")

            # train the Q-network
            if i % self.args['testInterval'] == 0:
                np.random.shuffle(trainingData)
                self.qnet.train(trainingData)
                self.qnet.saveModel(folder=self.args['checkpoint'], filename='temp.tar')

            # save the training history
            self.trainingHistory.extend(trainingDataToStore)
            if len(self.trainingHistory) > self.args['trainingHistoryMaxLen']:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainingHistory)}")
                self.trainingHistory = self.trainingHistory[-self.args['trainingHistoryMaxLen']:]
            
            # save the training history to checkpoint folder
            if i % self.args['testInterval'] == 0:
                self.saveTrainingHistory()

            # update the target network periodically
            # self.qnet.saveModel(folder=self.args['checkpoint'], filename='temp.tar')
            # if i % self.args['targetUpdatePeriod'] == 0:
            #     log.info('Updating target network...')
            #     self.targetNet.loadModel(folder=self.args['checkpoint'], filename='temp.tar')
            # Evaluate the agent performance
            if i % self.args['testInterval'] == 0:
                log.info('PLAYING AGAINST PREVIOUS VERSION')
                self.test_against([DQNPlayer(self.game, self.qnet).play, RandomPlayer(self.game).play, RandomPlayer(self.game).play], ["MODEL", "Random", "Random"])
                self.test_against([DQNPlayer(self.game, self.qnet).play, GreedyPlayer(self.game).play, RandomPlayer(self.game).play], ["MODEL", "Greedy", "Random"])

                player1 = DQNPlayer(self.game, self.qnet)
                player2 = GreedyPlayer(self.game)
                player3 = AdvancedGreedyPlayer(self.game)
                players = [player1.play,
                            player2.play,
                                player3.play]
                player_names = ["MODEL", "Greedy", "AdvancedGreedy"]

                playground = PlayGround(self.game, players, player_names, self.args['maxSteps'])
                performance = playground.playGames(self.args['arenaCompare'], display_flag=False, rotate_flag=True)
                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (performance[0]['winning'], performance[0]['losing'], performance[0]['draws']))
                log.info(performance)
                if performance[0]['winning'] + performance[0]['losing'] == 0 or float(performance[0]['winning']) / (performance[0]['winning'] + performance[0]['losing'] + performance[0]['draws']) < self.args['updateThreshold']:
                    log.info(f'Performance is not good enough. Continue training...')
                else:
                    log.info('Success trained the model! Save and exit...')
                    self.qnet.saveModel(folder=self.args['checkpoint'], filename=self.getModelName(i))
                    self.qnet.saveModel(folder=self.args['checkpoint'], filename='best.model.tar')
                    log.info('Training completed')
                    test = input("test")
                    break
        

    def getModelName(self, iteration):
        return 'checkpoint_' + str(iteration) + '_model.tar'

    def test_against(self, players, player_names):
        playground = PlayGround(self.game, players, player_names, self.args['maxSteps'])
        performance = playground.playGames(self.args['arenaCompare'], display_flag=False, rotate_flag=True)
        for player_name, player_performance in zip(player_names, performance):
            log.info(f"{player_name} wins: {player_performance['winning']}, loses: {player_performance['losing']}, draws(exceed max steps): {player_performance['draws']}")
        if performance[0]['winning'] + performance[0]['losing'] != 0 and float(performance[0]['winning']) / (performance[0]['winning'] + performance[0]['losing'] + performance[0]['draws']) >= self.args['updateThreshold']:
            beaten_model_name = 'winning.against.' + player_names[1] + '.tar'
            self.qnet.saveModel(folder=self.args['checkpoint'], filename=beaten_model_name)

    def saveTrainingHistory(self):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "trainingData.examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainingHistory)
        f.closed

    def checkFileExists(self, name):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            return False
        filename = os.path.join(folder, name)
        return os.path.exists(filename)



class SelfPlay():
    """
    This class is used for self-play training.
    inputs:
        game: the game class containing the game rules
        nnet: the neural network class as players
        args: the arguments for training
    outputs:
        training examples of (state, action, reward)
    """
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def play(self, agent, target_agent, epsilon, numEpisodes):
        """
        Runs {numEpisodes} episodes of self-play and returns the training data.
        """
        trainingData = []

        for _ in tqdm(range(numEpisodes), desc="Self Play"):
            trainingData.extend(self.runEpisode(agent, target_agent, epsilon))
        return trainingData

    def getRandomAction(self, valids):
        available_actions = []
        for i in range(len(valids)):
            if valids[i]:
                available_actions.append(i)
        return np.random.choice(available_actions)

    def runEpisode(self, agent, target_agent, epsilon):
        """
        Run a single game (a.k.a. episode) of self-play and return the training data.
        """
        trainDataPerGame = []
        state = self.game.getInitState()
        self.curPlayer = 1
        episodeStep = 0
        greedyPlayer = GreedyPlayer(self.game)
        advancedGreedyPlayer = AdvancedGreedyPlayer(self.game)
        
        while True:
            episodeStep += 1

            canonicalState = self.game.getCanonicalForm(state, self.curPlayer)
                        
            valids = self.game.getValidMoves(canonicalState, 1)
            
            q_values = agent.predict(canonicalState)[0]

            if self.curPlayer == 1:
                # model action
                if np.random.random() < epsilon:
                    action = self.getRandomAction(valids)
                else:
                    action = np.argmax(q_values*valids)
                    if valids[action] == 0:
                        action = self.getRandomAction(valids)
            elif self.curPlayer == 2:
                # greedy action
                action = advancedGreedyPlayer.play(canonicalState)
            else:
                # random action
                action = advancedGreedyPlayer.play(canonicalState)
            # print("player: ", self.curPlayer)
            # print("action: ", self.game.displayAction(action))
            pre_player = self.curPlayer
            state, self.curPlayer, reward = self.game.getNextState(state, self.curPlayer, action)
            # store the record
            trainDataPerGame.append((canonicalState, action, pre_player, q_values, reward))

            r = self.game.getGameEnded(state, self.curPlayer)
            if r[0] != 0 or episodeStep == self.args['maxSteps']:
                
                if episodeStep == self.args['maxSteps']:
                    r = [] # when exceeding max steps, make sure all players get -1 reward

                result = []

                # categorize the training data by player to find the next state for each player
                trainDataPerPlayer = [[], [], []]
                for canonicalState, action, player, q_values, reward in trainDataPerGame:
                    trainDataPerPlayer[player-1].append((canonicalState, action, q_values, reward))
                
                # update the q_values for each player
                for player in range(len(trainDataPerPlayer)):
                    for i in range(len(trainDataPerPlayer[player])):
                        canonicalState, action, q_values, reward = trainDataPerPlayer[player][i]

                        # calculate the maxQ value for the next state
                        # max_target_q_value = 0
                        # if i < len(trainDataPerPlayer[player]) - 1:
                        #     canonical_next_state = trainDataPerPlayer[player][i+1][0]
                        #     target_q_values = target_agent.predict(canonical_next_state)[0]
                        #     valids = self.game.getValidMoves(canonical_next_state, 1)
                        #     max_target_q_value = np.max(target_q_values*valids)
                        #     # q_values[action] = reward/self.args['rewardScale'] - self.args['penalty'] + max_target_q_value
                        #     q_values[action] = self.args['gamma']*max_target_q_value
                        # else:
                        #     q_values[action] = 1 if (player+1) in r else -1
                        # result.append((canonicalState, q_values.copy()))
                        if player > 0:
                            q_values = np.zeros(len(q_values))
                            q_values[action] = 1

                            result.append((canonicalState, q_values.copy()))
                    # print("new q_value: ", q_values[action])

                #     result.append((canonicalState, q_values))
                
                return result
