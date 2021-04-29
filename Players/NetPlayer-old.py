
## Net Player
import os
import os.path as path

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import random

from Players.Player import Player
from Network.Network import TronNet

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NetPlayer(Player):
    def __init__(self, model_name='default', logging=False):
        super(NetPlayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.net = TronNet()
        self.net = self.net.to(self.device)
        self.logging = logging

        self.optimiser = optim.Adam(self.net.parameters(), lr=0.001)
        self.action_probs_list = []
        self.action_rewards    = []
        self.eps   = np.finfo(np.float32).eps.item()
        self.epoch = 0
        self.depth = 5

        self.load_weights(model_name)

    def load_weights(self, _model_name):
        fname = path.join('models', _model_name)
        if os.path.exists(fname): 
            checkpoint = torch.load(fname)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Loaded with', self.epoch, 'epochs.')
        else: 
            print('weights not found for', _model_name)
    
    def save_weights(self, _model_name):
        _filename = path.join('models', _model_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }, _filename)
        print('Model saved.')
        
    def preprocess(self, _board, _location, _actions):
        proximity = np.array([self.depth] * len(_actions))
        for i in range(len(_actions)):
            for p in range(self.depth):
                if _board[tuple(_location + ((p+1) * _actions[i]) )] != 0:
                    proximity[i] = p
                    break
                
        proximity = torch.tensor(proximity).float()
        proximity = proximity.unsqueeze(dim=0)
        _board = torch.tensor(_board).unsqueeze(dim=0).unsqueeze(dim=0)
        return _board.to(self.device).float(), proximity.to(self.device)
    
    def get_action(self, _board, _location, _actions):
        board, dlc   = self.preprocess(_board, _location, _actions)
        probs, value = self.net(board, dlc)

        m      = Categorical(probs)
        action = m.sample()

        self.action_probs_list.append((m.log_prob(action), value))
        return action.item()

    def update_reward(self, _reward, _end_game):
        self.action_rewards.append(_reward)
    
        if _end_game:
            R = 0
            saved_actions = self.action_probs_list
            policy_losses = []
            values_losses = []
            returns       = []

            for r in self.action_rewards[::-1]:
                R = r + 0.95 * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 0.0001)

            for (log_prob, value), R in zip(self.action_probs_list, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                values_losses.append(
                    F.smooth_l1_loss(value, 
                                     torch.tensor([[R]]).to(self.device)))

            self.optimiser.zero_grad()
            loss = torch.stack(policy_losses).sum() +\
                   torch.stack(values_losses).sum()
            loss.backward()
            self.optimiser.step()

            self.action_probs_list = []
            self.action_rewards    = []
            self.epoch += 1
            if self.epoch % 1000 == 0:
                self.save_weights(self.model_name)