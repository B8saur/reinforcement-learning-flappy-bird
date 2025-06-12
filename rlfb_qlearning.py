import numpy as np
from rlfb_config import *
from engine import *
from collections import defaultdict
import random
from copy import copy
from tqdm import tqdm 

def get_discrete_state(game_state):
    game_state = copy(game_state)
    x_position, game_info, end_info = game_state
    bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
    current_pipe, status = end_info

    next_pipe_mid = (next_pipe_low + next_pipe_high)/2

    # print("BIRD_HEIGHT:", bird_height)
    # print("BIRD_SPEED_VERTICAL:", bird_speed_vertical)
    # print("NEXT_PIPE_DISTANCE:", next_pipe_distance)
    # print("NEXT_PIPE_MID:", next_pipe_mid)
    # Convert to more discrete state for Q learning
    bird_height = int(bird_height * 10)
    bird_speed_vertical = int(bird_speed_vertical * 200)
    next_pipe_distance = int(next_pipe_distance * 20)
    next_pipe_mid = int(next_pipe_mid * 20)
    
    bird_height = min(10, max(0,bird_height))

    # print("BIRD_HEIGHT:", bird_height)
    # print("BIRD_SPEED_VERTICAL:", bird_speed_vertical)
    # print("NEXT_PIPE_DISTANCE:", next_pipe_distance)
    # print("NEXT_PIPE_MID:", next_pipe_mid)

    # exit()
    return bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid, status



class Model_Q:
    def __init__(self):
        self.Q = defaultdict(lambda: [0, 0])

    def fit(self, episodes):
        cur_exploration_rate = q_exploration_rate
        for episode in tqdm(range(episodes), colour='green'):
            game = Game_engine()
            game_state = game.get_init_state()
            game_state = game.update(0) # TODO: do it correctly
            q_state = get_discrete_state(game_state)
            
            alive = True
            total_reward = 0


            while alive:
                if random.random() < cur_exploration_rate:
                    action = random.choice([0, 1]) # Moves randomly
                else:
                    action = int(self.Q[q_state][0] < self.Q[q_state][1])

                # Information unpacking
                next_game_state = game.update(action)
                next_q_state = get_discrete_state(next_game_state)

                bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid, alive = next_q_state
                reward = 1 if alive else -1000
                # Update Q table
                self.Q[q_state][action] += q_learning_rate * (reward + q_discount_factor * max(self.Q[next_q_state]) - self.Q[q_state][action])
                
                game_state = next_game_state
                q_state = next_q_state
                total_reward += reward
                
                # alive = status    
            cur_exploration_rate *= q_exploration_rate_decay
        
        print(f"Episode {episode}. total reward: ", total_reward)
    
    def action(self, game_state):
        q_state = get_discrete_state(game_state)
        return self.Q[q_state][0] < self.Q[q_state][1]
                