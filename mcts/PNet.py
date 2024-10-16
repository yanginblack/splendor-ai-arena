import os
import sys
sys.path.append('../')

import torch
import torch.optim as optim
import numpy as np
from .SplendorPNet import SplendorNNet as nnet
import time
from tqdm import tqdm
import torch.nn.functional as F

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
}

class NNetWrapper():
    def __init__(self, game):
        self.nnet = nnet(game, args)
        self.args = args
        if self.args['cuda']:
            print("CUDA is available")
            self.nnet.cuda()
        else:
            print("CUDA is not available")

    """
    Train the neural network model.
    """
    def train(self, train_data):
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args['epochs']):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            q_losses = AverageMeter()

            batch_count = int(len(train_data) / self.args['batch_size'])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(train_data), size=self.args['batch_size'])
                states, q_values = list(zip(*[train_data[i] for i in sample_ids]))

                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_q_values = torch.FloatTensor(np.array(q_values))

                # predict
                if self.args['cuda']:
                    states, target_q_values = states.contiguous().cuda(), target_q_values.contiguous().cuda()

                out_q_values = self.nnet(states)

                loss_q_values = self.loss_q_values(target_q_values, out_q_values)
                total_loss = loss_q_values

                # record loss
                q_losses.update(loss_q_values.item(), states.size(0))
                t.set_postfix(Loss_q=q_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def loss_q_values(self, targets, outputs):
        return F.mse_loss(targets, outputs)
        # return -torch.sum(targets * outputs) / targets.size()[0] # works for q_values as propabilities, required to be normalized

    def predict(self, state):
        """
        Predict the action probabilities of the state.
        return:
            q_values: a vector of q_values for actions
        """
        # construct input
        state = torch.FloatTensor(state.astype(np.float64))
        if self.args['cuda']:
            state = state.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            q_values = self.nnet(state)
        return q_values.data.cpu().numpy()
    
    def saveModel(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def loadModel(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

