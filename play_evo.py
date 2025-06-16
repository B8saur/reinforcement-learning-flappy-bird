import pygame
from pygame.locals import *

from game_config import *
import engine as eng
from drawable import *
import numpy as np
from evaluate import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("basic_ui_title")
clock = pygame.time.Clock()

##### paste your model here: (for models check file "models_evolutionary.py")
weights = [
    np.array([[ 2.27996565,  3.08299512, -1.43073265, -0.79306151,  3.58858851],
       [ 1.30464359, -0.98559919, -0.13745204,  1.67434462,  0.28692104],
       [-2.16037401, -2.06116002, -2.58359825,  1.77470963, -0.1749004 ],
       [-0.75041306, -1.62065008,  3.35709487, -0.76396582, -3.2975544 ],
       [ 0.17664216,  1.60120507,  0.7232882 ,  1.0949613 , -0.61289918]]),
    np.array([[ 1.3542234 , -0.11437597],
       [-0.26511151, -3.70767134],
       [-2.25991516,  0.47612996],
       [ 1.83055997,  1.26616871],
       [ 1.76549317, -1.13499047]])
]
biases = [
    np.array([[1.49010015, 0.24855471, 1.49358854, 1.14282114, 2.21020505]]),
    np.array([[0.57260984, 0.90622821]])
]
#####


# Game loop
run = True
while run:
    pipes = eng.get_pipes_list(HARD)
    engine = eng.Game_engine(pipes)
    first_pipe = 0

    x_position, data, result = engine.update(0)

    paused = False
    new_game = False
    while run and not new_game:
        clock.tick(60)

        jump = False
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):         # exit
                run = False
            elif e.type == KEYDOWN and e.key == K_SPACE:            # jump
                jump = True
            elif e.type == KEYDOWN and e.key == K_r:            # restart
                new_game = True
            elif e.type == KEYDOWN:         # pause and unpause
                paused = not paused


        for i in range(len(weights)):
            data = data @ weights[i] + biases[i]
        if(np.argmax(data) == 1):
            jump = True

        if not paused:
            x_position, data, result = engine.update(jump)
        

        screen.fill(BLUE)
        draw_circle(screen, data[0])
        for i in range(first_pipe, len(pipes)):
            if(pipes[i].position + pipes[i].length + LEFT_BUFFER < x_position):         # already behind
                first_pipe += 1
                continue
            if(pipes[i].position > x_position + WIDTH):           # too far ahead
                break
            draw_pipe(screen, x_position, pipes[i])
        screen.blit(pygame.transform.flip(screen, False, True), (0,0))
        draw_text(screen, "({:0.3f})    ({:0.3f},{:0.3f},{:0.3f},{:0.3f},{:0.3f})    ({:d},{:d})".
                format(loss(x_position,data), data[0], data[1], data[2], data[3], data[4], result[0], result[1]))


        pygame.display.update()
