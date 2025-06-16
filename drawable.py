import pygame
from pygame.locals import *

from game_config import *

class Rectangle(pygame.sprite.Sprite):
    def __init__(self, a, b):
        super().__init__()
        
        self.surf = pygame.Surface((a, b))
        self.surf.fill(GREEN)

def draw_pipe(screen, x_position, pipe):
    up = Rectangle(pipe.length*HEIGHT, (pipe.middle - pipe.radius)*HEIGHT)
    down = Rectangle(pipe.length*HEIGHT, (1 - pipe.middle - pipe.radius)*HEIGHT)

    screen.blit(up.surf, ((pipe.position - x_position + LEFT_BUFFER)*HEIGHT, 0))
    screen.blit(down.surf, ((pipe.position - x_position + LEFT_BUFFER)*HEIGHT, (pipe.middle + pipe.radius)*HEIGHT))

def draw_circle(screen, y_position):
    pygame.draw.circle(screen, YELLOW, (LEFT_BUFFER*HEIGHT, y_position*HEIGHT), B_RADIUS*HEIGHT)

def draw_text(screen, txt):
    font = pygame.font.Font('freesansbold.ttf', int(HEIGHT/13))
    text = font.render(txt, True, RED)
    textRect = text.get_rect()
    textRect.center = (WIDTH/2, HEIGHT/10)
    screen.blit(text, textRect)
