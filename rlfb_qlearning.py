import numpy as np
from game_config import *
from learning_config import *
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

    bird_height = int(bird_height * 15)
    bird_speed_vertical = int(bird_speed_vertical * 300)
    next_pipe_distance = int(next_pipe_distance * 20)
    next_pipe_mid = int(next_pipe_mid * 20)
    
    bird_height = min(15, max(0,bird_height))
    bird_speed_vertical = min(5, max(-5,bird_speed_vertical))

    # print(bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid)

    q_state = bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid
    return q_state, status

def get_better_discrete_state(game_state):
    game_state = copy(game_state)
    x_position, game_info, end_info = game_state
    bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
    current_pipe, status = end_info

    next_pipe_mid = (next_pipe_low + next_pipe_high)/2

    next_pipe_relative_mid_dist = (next_pipe_mid - bird_height)

    bird_height = int(bird_height * 15)
    bird_speed_vertical = int(bird_speed_vertical * 300)
    next_pipe_distance = int(next_pipe_distance * 20)
    next_pipe_mid = int(next_pipe_mid * 20)
    next_pipe_relative_mid_dist = int(next_pipe_relative_mid_dist * 10)
    
    # next_pipe_distance = min(15, max(0,next_pipe_distance))
    bird_height = min(15, max(0,bird_height))
    bird_speed_vertical = min(5, max(-5,bird_speed_vertical))
    next_pipe_relative_mid_dist = min(10, max(-10, next_pipe_relative_mid_dist))

    q_state = bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_relative_mid_dist
    return q_state, status

class Model_Q:
    def __init__(self):
        self.Q = defaultdict(lambda: [0, 0])

    def fit(self, episodes):
        cur_exploration_rate = q_exploration_rate
        for episode in tqdm(range(episodes), colour='green'):
            pipes = get_pipes_list(True, PIPE_COUNT_LEARN)
            game = Game_engine(pipes)
            game_state = game.update(0)
            q_state, _ = get_discrete_state(game_state)
            
            alive = True
            total_reward = 0


            while alive:
                if random.random() < cur_exploration_rate:
                    action = random.choice([0, 1]) # Moves randomly
                else:
                    action = np.argmax(self.Q[q_state]) 

                # Information unpacking
                next_game_state = game.update(action)
                next_q_state, alive = get_discrete_state(next_game_state)

                x_position, game_info, end_info = game_state
                bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
                next_pipe_mid = (next_pipe_low + next_pipe_high) / 2

                # reward = 1 if alive else -1000
                reward = 1
                if not alive:
                    reward = -1000
                elif abs(bird_height - next_pipe_mid) < 0.1:
                    reward += 1
                
                # Update Q table
                self.Q[q_state][action] += q_learning_rate * (reward + q_discount_factor * max(self.Q[next_q_state]) - self.Q[q_state][action])
                
                game_state = next_game_state
                q_state = next_q_state
                total_reward += reward
                
                # alive = status    
            cur_exploration_rate = max(q_exploration_rate_min, cur_exploration_rate * q_exploration_rate_decay)
        
        print(f"Episode {episode}. total reward: ", total_reward)
    
    def action(self, game_state):
        q_state, _ = get_discrete_state(game_state)
        return np.argmax(self.Q[q_state]) 
            
    def action_evaluate(self, game_info):
        dummy_game_state = (0, game_info, (0,0))
        return self.action(dummy_game_state)

class Model_TD:
    def __init__(self):
        self.Q = defaultdict(lambda: [0, 0])

    def fit(self, episodes):
        cur_exploration_rate = q_exploration_rate
        for episode in tqdm(range(episodes), colour='green'):
            pipes = get_pipes_list(True, PIPE_COUNT_LEARN)
            game = Game_engine(pipes)
            game_state = game.update(0)
            q_state, _ = get_discrete_state(game_state)

            alive = True
            total_reward = 0

            if random.random() < cur_exploration_rate:
                action = random.choice([0, 1])
            else:
                action = np.argmax(self.Q[q_state]) 

            while alive:
                # Information unpacking
                next_game_state = game.update(action)
                next_q_state, alive = get_discrete_state(next_game_state)

                # reward = 1 if alive else -1000
                x_position, game_info, end_info = game_state
                bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
                next_pipe_mid = (next_pipe_low + next_pipe_high) / 2

                # reward = 1 if alive else -1000
                reward = 1
                if not alive:
                    reward = -1000
                elif abs(bird_height - next_pipe_mid) < 0.1:
                    reward += 1
                
                if random.random() < cur_exploration_rate:
                    next_action = random.choice([0, 1])
                else:
                    next_action = np.argmax(self.Q[next_q_state]) 

                # Update Q table
                self.Q[q_state][action] += q_learning_rate * (reward + q_discount_factor * self.Q[next_q_state][next_action] - self.Q[q_state][action])
                
                game_state = next_game_state
                q_state = next_q_state
                action = next_action
                total_reward += reward
                
                # alive = status    
            cur_exploration_rate = max(q_exploration_rate_min, cur_exploration_rate * q_exploration_rate_decay)
        
        print(f"Episode {episode}. total reward: ", total_reward)
    
    def action(self, game_state):
        q_state, _ = get_discrete_state(game_state)
        return self.Q[q_state][0] < self.Q[q_state][1]
    
    def action_evaluate(self, game_info):
        dummy_game_state = (0, game_info, (0,0))
        return self.action(dummy_game_state)
    
# Better discretization
class Model_Q_v2:
    def __init__(self):
        self.Q = defaultdict(lambda: [0, 0])

    def fit(self, episodes):
        cur_exploration_rate = q_exploration_rate
        for episode in tqdm(range(episodes), colour='green'):
            pipes = get_pipes_list(True, PIPE_COUNT_LEARN)
            game = Game_engine(pipes)
            game_state = game.update(0)
            q_state, _ = get_discrete_state(game_state)
            
            alive = True
            total_reward = 0


            while alive:
                if random.random() < cur_exploration_rate:
                    action = random.choice([0, 1]) # Moves randomly
                else:
                    action = np.argmax(self.Q[q_state]) 

                # Information unpacking
                next_game_state = game.update(action)
                next_q_state, alive = get_better_discrete_state(next_game_state)

                x_position, game_info, end_info = game_state
                bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
                next_pipe_mid = (next_pipe_low + next_pipe_high) / 2

                # reward = 1 if alive else -1000
                reward = 1
                if not alive:
                    reward = -1000
                elif abs(bird_height - next_pipe_mid) < 0.1:
                    reward += 1
                
                # Update Q table
                self.Q[q_state][action] += q_learning_rate * (reward + q_discount_factor * max(self.Q[next_q_state]) - self.Q[q_state][action])
                
                game_state = next_game_state
                q_state = next_q_state
                total_reward += reward
                
                # alive = status    
            cur_exploration_rate = max(q_exploration_rate_min, cur_exploration_rate * q_exploration_rate_decay)
        
        print(f"Episode {episode}. total reward: ", total_reward)
    
    def action(self, game_state):
        q_state, _ = get_discrete_state(game_state)
        return np.argmax(self.Q[q_state])
     
    def action_evaluate(self, game_info):
        dummy_game_state = (0, game_info, (0,0))
        return self.action(dummy_game_state)