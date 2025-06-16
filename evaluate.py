import engine as eng
import numpy as np
from game_config import *

def loss(x_position, state):
    return x_position - np.abs(state[0] - (state[3]-state[2])/2)


def evaluate(func):
    repeats = 100

    result = 0
    for _ in range(repeats):
        pipes = eng.get_pipes_list(True, PIPE_COUNT_LEARN)
        game = eng.Game_engine(pipes)
        x_position, state, output = game.update()
        while(output[1]):
            decision = func(state)
            x_position, state, output = game.update(decision)

        result += loss(x_position, state)
    return result/repeats
    
