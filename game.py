import copy
import numpy as np
import settings as st
from gui import GUI


from live_plot import plot_durations

from Players.Player import BotPlayer
class Game:

    def __init__(self, players, use_gui=False):
        if use_gui:
            self.window = GUI()
        
        self.use_gui   = use_gui
        self.players   = [BotPlayer(), players]
        self.penlty    = ((st.MAP_SIZE-2)**2) //len(self.players)
        self.actions   = {0: np.array([-1,  0]),   ## UP
                          1: np.array([ 1,  0]),   ## DOWN
                          2: np.array([ 0, -1]),   ## LEFT
                          3: np.array([ 0,  1])}   ## RIGHT

        self.ssize = st.MAP_SIZE
        print('POST 0')
        print('MAX PENALTY', self.penlty)
        self.pstate = np.zeros((3, self.ssize, self.ssize))
        self.pview  = np.zeros((3, 3))


    def get_state(self):
        self.pview[::]  = 0
        x, y = self.positions[1]
        self.pview = copy.deepcopy(self.board[ x-1:x+2, y-1:y+2 ])
        self.pview = np.clip(self.pview, 0, 1)

        self.pstate[::] = 0   ## 1 is the bot player pos
        self.pstate[0][tuple(self.positions[1])] = 1
        self.pstate[1] = self.board.copy()
        self.pstate[1][tuple(self.positions[1])] = 0
        self.pstate[1][tuple(self.positions[0])] = 1
        self.pstate[2][tuple(self.positions[0])] = 1
        return (copy.deepcopy(self.pstate), copy.deepcopy(self.pview))

    def reset(self):
        interval = st.MAP_SIZE // (len(self.players) + 1)
        midpoint = st.MAP_SIZE // 2

        # p = np.arange(st.MAP_SIZE-2) - (midpoint-1)
        #  + np.random.choice(p, 1)[0]
        self.positions = [ np.array([(k+1) * interval, midpoint,]) \
                                for k in range(len(self.players))]
        
        self.board = np.ones((st.MAP_SIZE, st.MAP_SIZE))
        ## Set boarder to 1 (as obstacle)
        self.board[ 1:st.MAP_SIZE-1, 1:st.MAP_SIZE-1] = 0
        ## Set player heads
        for k in range(len(self.players)):
            self.board[tuple(self.positions[k])] = k+2

        self.penlty    = ((st.MAP_SIZE-2)**2) //len(self.players)

    ## Player: player index, 
    ## action: action index
    def move(self, player, action):
        valid = self.valid(player, action)
        destination = self.positions[player] + self.actions[action]
        self.board[tuple(self.positions[player])] = 1
        self.board[tuple(destination)] = player+2
        self.positions[player] = destination
        return valid

    def valid(self, player, action):
        destination = self.positions[player] + self.actions[action]
        if self.board[tuple(destination)] != 0: return False
        return True

    def play(self, ifp):
        game_steps = 0
        game_end = False
        moves = [None] * len(self.players)
        valid = [True] * len(self.players)
        score = [None] * len(self.players)
        while not(game_end):
            for i, player in enumerate(self.players):
                dstate = {'bot': (self.board, self.positions[i], self.actions),
                          'net':  self.get_state()}
                action   = player.get_action(dstate)
                moves[i] = action
                valid[i] = self.valid(i, action)
                score[i] = 1 if valid[i] else -self.penlty

                ## Some dude killed it self. 
                if valid[i] == False: game_end = True

            for player, action in enumerate(moves):
                self.move(player, action)

            ## if its a draw
            if np.array_equal(self.positions[1], self.positions[0]):
                game_end=True
                score[0] = -self.penlty//2
                score[1] = -self.penlty//2
                valid = [False, False]
             
            ## if player wins
            if valid == [False, True]:
                score[1] = self.penlty
            if valid == [True, False]:
                score[0] = self.penlty

            for i in range(2):
                self.players[i].update_reward(self.get_state(), score[i], game_end)

            game_steps += 1
            # self.penlty-= 1
            if self.use_gui: 
                # print('REWARDS:', score)
                # print(self.board)
                # print()
                # # print(self.get_state())
                # print('\n\n')
                self.window.update_frame(copy.deepcopy(self.board))
        if self.use_gui: 
            plot_durations(self.players[1].episode_rewards)
        if ifp:
            print('Game Steps:', f'{game_steps: >4}', 'Winners:', valid, 
                'Stock Market:', self.players[1].episode_rewards[-1])