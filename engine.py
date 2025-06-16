import numpy as np
from game_config import *

class Pipe:
    def __init__(self, prev_pos, radius = PIPE_RADIUS, length = PIPE_LENGTH):
        self.middle = np.random.uniform(PIPE_LOW, PIPE_HIGH)
        self.length = length
        self.radius = np.random.normal(radius, PIPE_RADIUS_DEVIATION)

        self.position = prev_pos + np.random.normal(INTERPIPE_DISTANCE, INTERPIPE_DEVIATION)

def get_pipes_list(learn = False, pipe_count = PIPE_COUNT):
    result = [None] * pipe_count
    result[0] = Pipe(INTRO_DISTANCE)

    radius = PIPE_RADIUS
    decrease = PIPE_RADIUS_DECREASE
    if learn:
        decrease = PIPE_RADIUS_DECREASE_LEARN
    
    for i in range(1, pipe_count):
        radius -= decrease
        result[i] = Pipe(result[i-1].position, radius)
    
    return result

class Game_engine:
    def __init__(self, pipes):
        self.x_position = 0
        self.y_position = INTRO_X_POS
        self.vertical_speed = 0
        self.current_pipe = 0
        self.state = True

        self.pipes = pipes
    
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

            if self.current_pipe == len(self.pipes):
                self.state = False
                self.current_pipe -= 1

            if self.detect_collision():
                self.state = False

        return self.x_position, \
            (self.y_position, self.vertical_speed, \
                self.pipes[self.current_pipe].middle - self.pipes[self.current_pipe].radius, \
                self.pipes[self.current_pipe].middle + self.pipes[self.current_pipe].radius, \
                self.pipes[self.current_pipe].position - self.x_position - B_RADIUS), \
            (self.current_pipe, self.state)

        # should return:
        # x_position
        # (bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance)
                # input for rl algorithm
                # last one is negative when inside the pipe 
        # (current_score (int), status)
                # status=False means lost, should stop and evaluate the result

    def detect_collision(self):
        if self.y_position + B_RADIUS > 1:          # top
            return True
        if self.y_position - B_RADIUS < 0:          # bottom
            return True

        # pipe collision detection:
        if self.x_position + B_RADIUS < self.pipes[self.current_pipe].position:
            return False            # current pipe is to far away to hit it
        
        range = B_RADIUS
        cur_pipe = self.pipes[self.current_pipe]

        if self.x_position < cur_pipe.position:            # mid before pipe
            range = np.sqrt(B_RADIUS**2 - (self.x_position - cur_pipe.position)**2)
        elif self.x_position > cur_pipe.position + cur_pipe.length:
            range = np.sqrt(B_RADIUS**2 - (self.x_position - cur_pipe.position - cur_pipe.length)**2)
        
        if cur_pipe.middle + cur_pipe.radius < self.y_position + range:
            return True         # upper half of the bird hit the pipe
        if cur_pipe.middle - cur_pipe.radius > self.y_position - range:
            return True         # upper half of the bird hit the pipe       
        return False
    