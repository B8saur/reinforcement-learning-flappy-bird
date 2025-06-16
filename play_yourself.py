import pygame
from pygame.locals import *

from game_config import *
import engine as eng
from drawable import *
from evaluate import *


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("basic_ui_title")
clock = pygame.time.Clock()


# Game loop
run = True
while run:
    pipes = eng.get_pipes_list()            # ADD "True" AS AN ARGUMENT HERE
    engine = eng.Game_engine(pipes)
    first_pipe = 0
    x_position = 0

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

        if not paused:
            x_position, data, result = engine.update(jump)
            print(loss(x_position, data))


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
        draw_text(screen, "({:0.3f},{:0.3f},{:0.3f},{:0.3f},{:0.3f})    ({:d},{:d})".
                format(data[0], data[1], data[2], data[3], data[4], result[0], result[1]))


        pygame.display.update()
