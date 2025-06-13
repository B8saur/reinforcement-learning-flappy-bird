# screen size
WIDTH = 800
HEIGHT = 450

# colors
RED = (255, 27, 9)
GREEN = (138, 173, 73)
BLUE = (78, 192, 202)
YELLOW = (249, 241, 42)

# game parameters
LEFT_BUFFER = 0.2
INTRO_X_POS = 0.5
INTRO_DISTANCE = 0.6
PIPE_COUNT = 20

# pipe parameters
PIPE_RADIUS = 0.25
PIPE_LOW = 0.3
PIPE_HIGH = 0.7
PIPE_LENGTH = 0.35

# pipe positioning
INTERPIPE_DISTANCE = 0.9
INTERPIPE_DEVIATION = 0.1

# bird stats
B_RADIUS = 0.05
SPEED = 0.007
GRAVITY = 0.0007
MAX_V_SPEED = 0.015

# Q-learning parameters
q_learning_rate = 0.1
q_discount_factor = 0.9
q_exploration_rate = 0.995
q_exploration_rate_min = 0.001
q_exploration_rate_decay = 0.995

# NN config
max_grad_norm = 5.0