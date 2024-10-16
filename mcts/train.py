import sys
sys.path.append('..')

import logging
import coloredlogs
import os
from Splendor.SplendorGame import SplendorGame as Game
from NNet import NNetWrapper as nn
from Coach import Coach

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
    'cpuct': 2,
    'num_simulations': 80,
    'player_count': 3,
    'numIters': 2000,
    'episodes': 1000,
    'ubc_epsilon': 0.02,
    'iterQueueMaxLen': 20000,
    'trainingDataMaxLen': 10000,
    'arenaCompare': 30,
    'updateThreshold': 0.8,
    'maxSteps': 500,
    'maxDepth': 100,
    'checkpoint': './checkpoints/',
    'evaluation_interval': 20,
    'temp_file': 'temp.pth.tar',
    'best_file': 'best.pth.tar'
}

def main():
    log.info('Loading %s...', Game.__name__)
    game = Game()
    
    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()

