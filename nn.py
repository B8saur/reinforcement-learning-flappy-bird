import numpy as np
from learning_config import *
from engine import *
import random
from collections import deque
from copy import copy, deepcopy
from tqdm import tqdm 
import pickle 

def leaky_relu(x):
    return np.where(x>0, x, 0.1*x)

def leaky_relu_dx(x):
    return np.where(x>0, 1.0, 0.1)

def get_useful_state(game_state):
    game_state = copy(game_state)
    x_position, game_info, end_info = game_state
    bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
    current_pipe, status = end_info

    next_pipe_mid = (next_pipe_low + next_pipe_high)/2

    # Clamp values to avoid weird scenarios

    bird_height = np.clip(bird_height, 0, 1)
    bird_speed_vertical = np.clip(bird_speed_vertical, -1, 1)
    next_pipe_distance = np.clip(next_pipe_distance, -1, 1) # This is not true, BUT only for the first pipe
    next_pipe_mid = np.clip(next_pipe_mid, 0, 1)

    q_state = bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid
    return q_state, status

class Model_NN:
    def __init__(self):
        self.lr = 0.0001
        self.layer_sizes = [4, 16, 16, 2]
        self.replay = deque(maxlen=50000)
        self.batch_size = 64

        self.weights = []
        self.biases = []

        self.target_model = deepcopy(self)

        for i in range(len(self.layer_sizes)-1):
            num_in, num_out = self.layer_sizes[i], self.layer_sizes[i+1]
            w = np.random.randn(num_out, num_in) * np.sqrt(2/(num_in+num_out)) 

            b = np.zeros((num_out, 1))
            self.weights.append(w)
            self.biases.append(b)

    def update_target_model(self):
        self.target_model.weights = [w.copy() for w in self.weights]
        self.target_model.biases = [b.copy() for b in self.biases]

    def save_model(self, path = "model.pickle"):
        with open(path, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)

    def load_model(self, path = "model.pickle"):
        with open(path, 'rb') as file:
            self.weights, self.biases = pickle.load(file)

    def forward_pass(self, input):
        a = np.array(input, dtype=np.float32).reshape(-1, 1)
        activations = [a]
        z_cache = []

        for i in range(len(self.weights)-1):
            z = self.weights[i] @ a + self.biases[i]
            z_cache.append(z)
            a = leaky_relu(z)
            activations.append(a)

        # Output layer
        z = self.weights[-1] @ a + self.biases[-1]
        z_cache.append(z)
        activations.append(z)

        return activations[-1], activations, z_cache
    
    def backward_pass(self, activations, z_cache, target_vec, action_index):
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        dz = np.zeros_like(activations[-1])
        dz[action_index, 0] = activations[-1][action_index, 0] - target_vec[action_index, 0]
        dA = dz

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grad_w[i] = dA @ a_prev.T
            grad_b[i] = dA
            if i > 0:
                z = z_cache[i-1]
                da = self.weights[i].T @ dA
                dA = da * leaky_relu_dx(z)
        
        # Update
        for i in range(len(self.weights)):
            grad_w[i] = np.clip(grad_w[i], -100, 100)
            grad_b[i] = np.clip(grad_b[i], -100, 100)

            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i] -= self.lr * grad_b[i]

            self.weights[i] = np.clip(self.weights[i], -100, 100)
            self.biases[i] = np.clip(self.biases[i], -100, 100)

    # Save state, action, reward, action2 for replay
    def remember(self, s, a, r, s2, alive):
        self.replay.append((s, a, r, s2, alive))

    def train(self):
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)

        for s, a, r, s2, alive in batch:
            q_vals, activations, z_cache = self.forward_pass(s)
            # next_q_vals, _, _ = self.forward_pass(s2)
            next_q_vals, _, _ = self.target_model.forward_pass(s2) # Important - use the target_model instead of the current one
            
            target_q = r if not alive else r + 0.99 * np.max(next_q_vals)
            target_vec = q_vals.copy()
            target_vec[a, 0] = target_q
            self.backward_pass(activations, z_cache, target_vec, a)

    def action(self, game_state):
        state, alive = get_useful_state(game_state)
        out_val, _, _ = self.forward_pass(state)
        # if random.random() < 0.0005:
        #     print(out_val)
        return int(np.argmax(out_val))
    
    def fit(self, episodes):
        scores = []
        epsilon = 0.9
        for episode in tqdm(range(1,episodes+1), colour='green'):
            score = 0
            
            epsilon = max(0.001, epsilon * 0.999)
            game = Game_engine()
            game_state = game.get_init_state()
            game_state = game.update(0)
            state, alive = get_useful_state(game_state)

            while alive:
                if random.random() < epsilon:
                    action = random.choice([0, 1])
                else:
                    action = self.action(game_state)

                next_game_state = game.update(action)
                next_state, alive = get_useful_state(next_game_state)
                # reward = 1 if alive else -10
                if not alive:
                    reward = -10
                else:
                    x_position, game_info, end_info = game_state
                    bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
                    next_pipe_mid = (next_pipe_low + next_pipe_high) / 2

                    reward = 1 - abs(bird_height - next_pipe_mid) # Encourage staying close to the pipe gap

                self.remember(state, action, reward, next_state, alive)

                game_state = next_game_state
                state = next_state
                score += reward
            scores.append(score)
            self.train()
            
            if episode % 50 == 0:
                print(f"Avg reward (last 50 eps): {np.mean(scores[-50:])}")

            if episode % 500 == 0:
                self.update_target_model()
                self.save_model()