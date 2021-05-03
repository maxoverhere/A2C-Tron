## Base class player
## How it works.. 
## Look at each direction and take the longest route if possible
import numpy as np

## Inheritance Class
class Player():
    def __init__(self):
        self.episode_rewards = []
        pass

    def get_action(self, _board, _location, _actions):
        pass

    def update_reward(self, _state, _reward, _end_game):
        pass

## Proximity Bot
class BotPlayer(Player):
    def __init__(self, depth=2):
        super(BotPlayer, self).__init__()
        assert depth >= 1
        self.depth = depth
        print('Bot using depth:', depth)
        
    def get_action(self, _dstate):
        _board, _location, _actions = _dstate['bot']
        proximity = np.array([self.depth] * len(_actions))
        for i in range(len(_actions)):
            for p in range(self.depth):
                if _board[tuple(_location + (p+1) * _actions[i])] != 0:
                    proximity[i] = p
                    break
        
        # print("Depths", proximity)
        max_depth = max(proximity)
        max_actions = np.arange(len(_actions))[ proximity == max_depth ]
        return np.random.choice(max_actions, 1, replace=False)[0]
