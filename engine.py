import numpy as np
from rlfb_config import *

class Pipe:
    def __init__(self, prev_pos):
        self.middle = np.random.uniform(PIPE_LOW, PIPE_HIGH)
        self.length = PIPE_LENGTH
        self.radius = PIPE_RADIUS

        self.position = prev_pos + np.random.normal(INTERPIPE_DISTANCE, INTERPIPE_DEVIATION)


class Game_engine:
    def __init__(self, pipe_count = PIPE_COUNT):
        self.x_position = 0
        self.y_position = INTRO_X_POS
        self.vertical_speed = 0
        self.current_pipe = 0
        self.state = True

        self.pipes = [None] * pipe_count
        self.pipes[0] = Pipe(INTRO_DISTANCE)
        for i in range(1, pipe_count):
            self.pipes[i] = Pipe(self.pipes[i-1].position)

    def get_init_state(self):
        return self.pipes
    
    def update(self, jump = False):
        if self.state:            # game still going
            if jump:
                self.vertical_speed = MAX_V_SPEED
            self.x_position += SPEED
            self.y_position += self.vertical_speed
            self.vertical_speed -= GRAVITY

            while self.current_pipe < len(self.pipes) and \
                self.pipes[self.current_pipe].position + self.pipes[self.current_pipe].length \
                    < self.x_position - B_RADIUS:
                self.current_pipe += 1

            if(self.current_pipe == len(self.pipes)):
                self.state = False
                self.current_pipe -= 1

            # TODO: add colision detection

        return self.x_position, \
            (self.y_position, self.vertical_speed, \
                self.pipes[self.current_pipe].middle - self.pipes[self.current_pipe].radius, \
                self.pipes[self.current_pipe].middle + self.pipes[self.current_pipe].radius, \
                max(0, self.pipes[self.current_pipe].position - self.x_position - B_RADIUS)), \
            (self.current_pipe, self.state)

        # should return:
        # x_position
        # (bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance)
                # input for rl algorithm
                # last one is zero when inside the pipe 
        # (current_score (int), status)
                # status=False means lost, should stop and evaluate the result
