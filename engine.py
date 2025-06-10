import numpy as np
from rlfb_config import *

class Pipe:
    def __init__(self, prev_pos):
        self.middle = np.random.randint(PIPE_LOW, PIPE_HIGH)
        self.length = PIPE_LENGTH
        self.radius = PIPE_RADIUS

        self.position = prev_pos + np.random.normal(INTERPIPE_DISTANCE, INTERPIPE_DEVIATION)


class Game_engine:
    def __init__(self, pipe_count = PIPE_COUNT):
        self.pipes = [None] * pipe_count
        self.pipes[0] = Pipe(INTRO_DISTANCE)
        for i in range(1, pipe_count):
            self.pipes[i] = Pipe(self.pipes[i-1].position)

    def get_init_state(self):
        return self.pipes
    
    def update(jump = False):
        # should return:
        # x_position
        # (bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance)
                # input for rl algorithm
                # last one is zero when inside the pipe 
        # (current_score (int), status)
                # status=False means lost, should stop and evaluate the result
        return 0