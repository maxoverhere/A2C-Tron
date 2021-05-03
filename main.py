from Players.Player import Player, BotPlayer
from Players.NetPlayer import NetPlayer
from Players.DQNPlayer import DQNPlayer
from game import Game
from Options import Options
import settings as s

from datetime import datetime
import os

OPTIONS = Options()
OPTIONS = OPTIONS.parse()

def main():
    nplayer0 = None
    if OPTIONS.model_type=='A2C':
        nplayer0 = NetPlayer(OPTIONS.model_name)
    elif OPTIONS.model_type=='DQN':
        nplayer0 = DQNPlayer(OPTIONS.model_name)
    else:
        print(OPTIONS.model_type + ' Model not defined!')
        exit()
  
    Game_env = Game(nplayer0, use_gui=OPTIONS.gui, depth=OPTIONS.depth)
    epochs = 0
    k = False
    while(epochs < OPTIONS.epochs):
        epochs += 1
        k = epochs % 100 == 0
        if k:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('[' + current_time + ']', f"{epochs:08d} ", end='')
        Game_env.reset()
        Game_env.play(k)

if __name__ == "__main__":
    main()