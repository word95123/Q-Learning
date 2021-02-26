import random
import copy
from collections import deque
import numpy as np
class ER:
    def __init__(self, max_len = 100):
        #self.replay = deque(maxlen = max_len)
        self.replay = []
        self.max_len = max_len
        self.curr_game = []

    def store(self, state, action, reward, next_state):
        #self.replay.append((state, action, reward, next_state))
        self.curr_game.append((state, action, reward, next_state))

        discount = 0.9
        if reward != 0:
            r_0 = 0
            for s, a, r, _s in reversed(self.curr_game):
                r_0 = r + discount * r_0
                self.replay.append((s, a, r_0, _s))

            while len(self.replay) > self.max_len:
                self.replay.pop(0)
            self.curr_game = []

    
    def get_random_minibatch(self, size = 5):
        #idx = np.random.choice(np.arange(len(self.replay)), size = size)
        #return [self.replay[i] for i in idx]
        
        if len(self.replay) > size:
            copied = copy.deepcopy(self.replay)
            random.shuffle(copied)
            return copied[:size]
        else:
            return self.replay
        