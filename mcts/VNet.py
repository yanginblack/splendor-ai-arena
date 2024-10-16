import os
import sys
sys.path.append('../')

import torch
import torch.optim as optim
import numpy as np
from .SplendorVNet import SplendorVNet as nnet
import time
from tqdm import tqdm

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
}

class NNetWrapper():
    """
    This class is a wrapper for the neural network model.
    """
    def __init__(self, game):
        self.nnet = nnet(game, args)
        self.channels, self.rows, self.cols = game.getStateSize()
        self.action_size = game.getActionSize()

        if args['cuda']:
            print("CUDA is available")
            self.nnet.cuda()
        else:
            print("CUDA is not available")

    def train(self, examples):
        """
        Train the neural network model.
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args['epochs']):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            # pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args['batch_size'])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args['batch_size'])
                # states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                # target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args['cuda']:
                    # states, target_pis, target_vs = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                    states, target_vs = states.contiguous().cuda(), target_vs.contiguous().cuda()
                out_v = self.nnet(states)

                # l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                # total_loss = l_pi + l_v

                # record loss
                # pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                t.set_postfix(Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                # total_loss.backward()
                l_v.backward()
                optimizer.step()

    def predict(self, state):
        """
        Predict the action probabilities of the state.
        return:
            probs: a policy vector of probability distribution for actions
            value: a scalar value of game score
        """
        start = time.time()

        # construct input
        state = torch.FloatTensor(state.astype(np.float64))
        if args['cuda']:
            state = state.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            v = self.nnet(state)
        # return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        return v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if args['cuda'] else 'cpu'
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