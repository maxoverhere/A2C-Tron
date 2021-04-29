from Players.Player import Player, BotPlayer
from Players.NetPlayer import NetPlayer
from game import Game
from Options import Options
import settings as s

from datetime import datetime
import os

OPTIONS = Options()
OPTIONS = OPTIONS.parse()

def main():

    nplayer0 = NetPlayer('default_16')
    # Game_env = Game([nplayer0, BotPlayer()], use_gui=OPTIONS.gui)
    
    Game_env = Game(nplayer0, use_gui=OPTIONS.gui)
    epochs = 0
    k = False
    while(True):
        epochs += 1
        k = epochs % 100 == 0
        if k:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('[' + current_time + ']', f"{epochs:08d} ", end='')
        Game_env.reset()
        Game_env.play(k)

main()