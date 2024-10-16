import sys
sys.path.append('..')

import logging
import coloredlogs
from Splendor.SplendorGame import SplendorGame as Game
from Train import Train
from model import modelWrapper
import torch

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
    'numIters': 100,
    'numEpisodes': 2000,
    'startEpsilon': 1.0,
    'minEpsilon': 0.05,
    'decayRate': 0.99,
    'trainingHistoryMaxLen': 5000000,
    'trainingDataMaxLen': 2000000,
    'testInterval': 20,
    'targetUpdatePeriod': 5,
    'updateThreshold': 0.7,
    'arenaCompare': 300,
    'maxSteps': 300,
    'checkpoint': './checkpoints/',
    'rewardScale': 100,
    'penalty': 0.01,
    'gamma': 0.95,
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
}


if __name__ == "__main__":
    game = Game()
    qnet = modelWrapper(game, args)
    train = Train(game, qnet, args)
    train.start()
