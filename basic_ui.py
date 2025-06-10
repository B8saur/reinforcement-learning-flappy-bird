import pygame
from pygame.locals import *

from rlfb_config import *
import engine as eng


pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("basic_ui_title")
position = 0

class Rectangle(pygame.sprite.Sprite):
    def __init__(self, a, b):
        super().__init__()
        
        self.surf = pygame.Surface((a, b))
        self.surf.fill(GREEN)

def draw_pipe(pipe):
    up = Rectangle(pipe.length, pipe.middle - pipe.radius)
    down = Rectangle(pipe.length, HEIGHT - pipe.middle - pipe.radius)

    screen.blit(up.surf, (pipe.position - position, 0))
    screen.blit(down.surf, (pipe.position - position, pipe.middle + pipe.radius))


clock = pygame.time.Clock()

engine = eng.Game_engine()
pipes = engine.get_init_state()
first_pipe = 0

# Game loop
run = True
while run:
    clock.tick(60)
    screen.fill(BLUE)
    
    for i in range(first_pipe, len(pipes)):
        if(pipes[i].position + pipes[i].length < position):         # already behind
            first_pipe += 1
            continue
        if(pipes[i].position > position + WIDTH):           # too far ahead
            break
        draw_pipe(pipes[i])

    for e in pygame.event.get():
        if e.type == QUIT or e.type == KEYDOWN:
            run = False
    pygame.display.flip()
    position += SPEED



