import numpy as np
import engine as eng
from tqdm import tqdm
from game_config import *
from copy import deepcopy
import matplotlib.pyplot as plt
from evaluate import *

    

def activation(val):
    return np.clip(val, 0)

def softmax(output_array):
    output_array = np.exp(output_array)
    return output_array/np.sum(output_array)


def eval_model(all_weights, all_biases, pipes):
    engine = eng.Game_engine(pipes)
    position, state, output = engine.update()
    while(output[1]):
        state = np.array(state)
        for i in range(len(all_weights)):
            state = state @ all_weights[i] + all_biases[i]
        position, state, output = engine.update(np.argmax(state))
    
    return loss(position, state)


MEAN = 0
VARIANCE = 0.1
EVO_CHANGE = 0.0001

def fit(epochs, player_count = 100):
    network_dimensions = [5, 5, 2]

    pool = [None]*player_count
    # init
    for i in range(player_count):
        all_weights = [None]*(len(network_dimensions)-1)
        all_biases = [None]*(len(network_dimensions)-1)
        for j in range(len(network_dimensions)-1):
            all_weights[j] = np.random.normal(MEAN, VARIANCE, (network_dimensions[j], network_dimensions[j+1]))
            all_biases[j] = np.random.normal(MEAN, VARIANCE, (1, network_dimensions[j+1]))
        pool[i] = [all_weights, all_biases]

    results = None
    general = np.zeros((epochs, 2))
    for i in tqdm(range(epochs)):
        pipes = eng.get_pipes_list(True, PIPE_COUNT_LEARN)
        # evaluate
        results = np.array(
            [eval_model(pool[j][0], pool[j][1], pipes) for j in range(player_count)]
        )
        
        # choose the fittest
        save = int(player_count/10)         # save top 10
        change = int(3*save)            # alter top 30, 3 models for each
        
        sorted = np.argsort(-results)
        temp = [None]*change
        for j in range(change):
            temp[j] = deepcopy(pool[sorted[j]])
        
        for j in range(save):
            pool[j] = temp[j]
        for j in range(change):
            pool[save+j+0*change] = deepcopy(temp[j])           # copy, then change
            for k in range(len(network_dimensions)-1):
                pool[save+j+0*change][0][k] += np.random.normal(MEAN, VARIANCE, (network_dimensions[k], network_dimensions[k+1]))
                pool[save+j+0*change][1][k] += np.random.normal(MEAN, VARIANCE, (1, network_dimensions[k+1]))
            pool[save+j+1*change] = deepcopy(temp[j])           # copy, then change
            for k in range(len(network_dimensions)-1):
                pool[save+j+1*change][0][k] += np.random.normal(MEAN, VARIANCE, (network_dimensions[k], network_dimensions[k+1]))
                pool[save+j+1*change][1][k] += np.random.normal(MEAN, VARIANCE, (1, network_dimensions[k+1]))
            pool[save+j+2*change] = deepcopy(temp[j])           # copy, then change
            for k in range(len(network_dimensions)-1):
                pool[save+j+2*change][0][k] += np.random.normal(MEAN, VARIANCE, (network_dimensions[k], network_dimensions[k+1]))
                pool[save+j+2*change][1][k] += np.random.normal(MEAN, VARIANCE, (1, network_dimensions[k+1]))

        # print(results[sorted[:6]])
        general[i][0] = np.average(results)
        general[i][1] = np.average(results[:save])


    repeats = 10
    results = np.zeros((player_count))
    for _ in tqdm(range(repeats)):
        pipes = eng.get_pipes_list(True, PIPE_COUNT_LEARN)
        for i in range(player_count):
            results[i] += eval_model(pool[i][0], pool[i][1], pipes)
    results /= 10


    best = np.argmax(results)
    print(results[best])
    return pool[best][0], pool[best][1], general

epochs = 400
players = 400

w, b, general = fit(epochs, players)
print(w)
print(b)

plt.plot(general[:, 0], "-r", label="average all")
plt.plot(general[:, 1], "-g", label="average top 10%")
plt.legend(loc="upper left")
plt.title(f"Evolution results, {epochs} epochs, {players}, players")
plt.savefig(f"report/evo_example_{epochs}_{players}.png")
