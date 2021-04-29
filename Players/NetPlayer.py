import os, random, math, copy
import os.path as path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import settings as s
from Players.Player import Player

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 3 * s.MAP_SIZE * s.MAP_SIZE

DIR = os.path.dirname(os.path.join( os.path.dirname( __file__ ))) + '\models'
EXT = '.hdd'
SFQ = 10000

def t(x): return torch.tensor(x, device=device).float()

# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, n_actions),
        )

        self.views = nn.Sequential(
            nn.Linear(9, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

        self.merge = nn.Sequential(
            nn.Linear(8, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
        )
    
    def forward(self, state):
        X, V = torch.flatten(t(state[0])), torch.flatten(t(state[1]))
        overview = self.model(X)
        navigate = self.views(V)
        combined = torch.cat([overview, navigate], dim=0)
        return F.softmax(self.merge(combined), dim=0)

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, X):
        X = torch.flatten(t(X[0]))
        return self.model(X)

class NetPlayer(Player):
    def __init__(self, model_name='default', logging=False):
        super(NetPlayer, self).__init__()

        self.actor = Actor(STATE_SIZE, 4).to(device)
        self.critic = Critic(STATE_SIZE).to(device)

        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

        self.gamma = 0.99
        self.memory = Memory()
        self.max_steps = 50

        self.steps = 0
        self.acc_reward = 0
        self.episode_rewards = []
        ### ============== OLD STUFF ================= ###
        self.model_name = model_name
        self.load_weights()

    def load_weights(self, fname=None):
        print('Trying')
        if fname == None:
            fname = path.join(DIR, self.model_name + EXT)
        if not path.isfile(fname):
            print('Nothing found bro!')
            return
        checkpoint = torch.load(fname)

        self.episode_rewards = checkpoint['episode_rewards']
        self.steps = checkpoint['steps_trained']
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        self.adam_actor.load_state_dict(checkpoint['actor_optimizer'])
        self.adam_critic.load_state_dict(checkpoint['critic_optimizer'])
        
        print('Loaded Model:', fname)
    
    def save_weights(self):
        fname = path.join(DIR, self.model_name + EXT)
        
        torch.save({
            'episode_rewards'   : self.episode_rewards,
            'actor_state_dict'  : self.actor.state_dict(),
            'critic_state_dict' : self.critic.state_dict(),
            'actor_optimizer'   : self.adam_actor.state_dict(),
            'critic_optimizer'  : self.adam_critic.state_dict(),
            'steps_trained'     : self.steps
        }, fname)
        print('Model saved.')

    def get_action(self, dstate):
        state = dstate['net']
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        self.last_state  = state
        self.last_dist   = dist
        self.last_action = action
        self.last_critic = self.critic(state)
        return action.item()

    def train(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))

        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma * q_val * (1.0 - done)
            q_vals[len(self.memory)-1 - i] = q_val

        advantage = torch.tensor(q_vals, device=device) - values
        
        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()

        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()

        self.memory.clear()

    def update_reward(self, dstate, _reward, _end_game):
        self.memory.add(self.last_dist.log_prob(self.last_action),
                        self.last_critic, _reward, _end_game)

        self.acc_reward += _reward
        self.steps += 1
        if _end_game or (self.steps % self.max_steps == 0):
            last_q_val = self.critic(dstate).item()
            self.train(last_q_val)
        if _end_game: 
            self.episode_rewards.append(self.acc_reward)
            self.acc_reward = 0
            if len(self.episode_rewards) % SFQ == 0:
                self.save_weights()
                

