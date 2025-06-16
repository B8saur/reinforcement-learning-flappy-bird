import numpy as np
from learning_config import *
from engine import *
import random
from collections import deque
from copy import copy
from tqdm import tqdm 

def leaky_relu(x):
    return np.where(x>0, x, 0.01*x)

def leaky_relu_dx(x):
    return np.where(x>0, 1.0, 0.01)

# def sigmoid(x):
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + np.exp(-x))

def get_useful_state(game_state):
    game_state = copy(game_state)
    x_position, game_info, end_info = game_state
    bird_height, bird_speed_vertical, next_pipe_low, next_pipe_high, next_pipe_distance = game_info
    current_pipe, status = end_info

    next_pipe_mid = (next_pipe_low + next_pipe_high)/2

    q_state = bird_height, bird_speed_vertical, next_pipe_distance, next_pipe_mid
    return q_state, status

class Model_NN:
    def __init__(self):
        self.lr = 0.0001
        self.layer_sizes = [4, 16, 8, 2]
        self.replay = deque(maxlen=50000)
        self.batch_size = 64

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes)-1):
            num_in, num_out = self.layer_sizes[i], self.layer_sizes[i+1]
            w = np.random.randn(num_out, num_in) * np.sqrt(2/(num_in+num_out)) 

            b = np.zeros((num_out, 1))
            self.weights.append(w)
            self.biases.append(b)

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
    
    def backward_pass(self, activations, z_cache, out_target, action_index):
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        out = activations[-1]
        delta = np.zeros_like(out)

        delta[action_index, 0] = out[action_index, 0] - out_target[action_index, 0]

        dA = delta

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
            grad_w[i] = np.clip(grad_w[i], -max_grad_norm, max_grad_norm)
            grad_b[i] = np.clip(grad_b[i], -max_grad_norm, max_grad_norm)

            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i] -= self.lr * grad_b[i]

            self.weights[i] = np.clip(self.weights[i], -1, 1)
            self.biases[i] = np.clip(self.biases[i], -1, 1)

    # Save state, action, reward, action2 for replay
    def remember(self, s, a, r, s2, alive):
        self.replay.append((s, a, r, s2, alive))

    def train(self):
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)

        for s, a, r, s2, alive in batch:
            # Get prediction for the current state
            q_vals, activations, z_cache = self.forward_pass(s)
            # Estimate value of the next state
            next_q_vals, _, _ = self.forward_pass(s2)
            
            target_q = r if not alive else r + 0.99 * np.max(next_q_vals)
            target_vec = q_vals.copy()
            target_vec[a, 0] = target_q
            self.backward_pass(activations, z_cache, target_vec, a)

    def action(self, game_state):
        state, alive = get_useful_state(game_state)
        out_val, _, _ = self.forward_pass(state)
        # print(out_val)
        return int(np.argmax(out_val))
    
    def fit(self, episodes):
        for episode in tqdm(range(episodes), colour='green'):
            epsilon = max(0.001, 1.0 - episode / episodes)
            pipes = get_pipes_list(True, PIPE_COUNT_LEARN)
            game = Game_engine(pipes)
            game_state = game.update(0)
            state, alive = get_useful_state(game_state)

            while alive:
                if random.random() < epsilon:
                    action = random.choice([0, 1])
                else:
                    action = self.action(game_state)

                next_game_state = game.update(action)
                next_state, alive = get_useful_state(next_game_state)
                reward = 1 if alive else -10
                self.remember(state, action, reward, next_state, alive)


                game_state = next_game_state
                state = next_state
            if episode % 5 == 0:
                self.train()
            