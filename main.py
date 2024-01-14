import pygame
import numpy as np

from gamelib import *
from dqn import DQN


# Parameters
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

FPS = 60

ACCELERATION = 1200
ANGULAR_ACCELERATION = 600

AIR_RESISTANCE = 0.5
FRICTION_COEFFICIENT = 700
ANGULAR_FRICTION = 200
MAX_ANGULAR_VELOCITY = 400

# Agents
# 0 => Player
# 1 => ML Agent (does not learn)
# 2 => ML Agent (that learns)
CRIMINAL_AGENT = 1
POLICE_AGENT = 0


# Library objects
clock = pygame.time.Clock()


# Global variables
fps = FPS
spf = 1 / fps

criminal: Car = None
police: Car = None


def main():
    global criminal, police

    pygame.init()

    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Car Simulation")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    fullscreen = False

    running = True

    inputs = InputHandler()

    criminal_sprite = pygame.image.load("red_car.png")
    criminal_sprite = pygame.transform.scale(criminal_sprite, (0.5 * criminal_sprite.get_width(), 0.5 * criminal_sprite.get_height()))
    spawnpoint = Vector(100, 100)
    criminal = Car(spawnpoint, criminal_sprite, ACCELERATION * spf, ANGULAR_ACCELERATION * spf, AIR_RESISTANCE, FRICTION_COEFFICIENT, ANGULAR_FRICTION)
    
    police_sprite = pygame.image.load("police_car.png")
    police_sprite = pygame.transform.scale(police_sprite, (0.5 * police_sprite.get_width(), 0.5 * police_sprite.get_height()))
    spawnpoint = Vector(200, 300)
    police = Car(spawnpoint, police_sprite, ACCELERATION * spf, ANGULAR_ACCELERATION * spf, AIR_RESISTANCE, FRICTION_COEFFICIENT, ANGULAR_FRICTION)


    # Setup DQN
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99


    while running:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle input
        inputs.update()
        inputs = inputs.get_global()

        if inputs.fullscreen_p:
            if fullscreen:
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                fullscreen = False  
            else:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                fullscreen = True

        if not (CRIMINAL_AGENT and POLICE_AGENT):
            actions = inputs.get_actions()
            player = criminal if not CRIMINAL_AGENT else police

            # Handle input
            if actions.forward:
                player.accelerate(1)
            if actions.backward:
                player.accelerate(-1)
            if actions.left:
                player.rotate(1)
            if actions.right:
                player.rotate(-1)

        # Update transform
        criminal.update(spf)
        police.update(spf)

        draw(screen)


def draw(screen):
    global criminal, police

    screen.fill((100, 100, 100))
    
    criminal.blit(screen)
    police.blit(screen)

    # Draws a cross on relevant points
    # pygame.draw.line(screen, (0, 255, 0), (player.position.x - 10, player.position.y), (player.position.x + 10, player.position.y), 2)
    # pygame.draw.line(screen, (0, 255, 0), (player.position.x, player.position.y - 10), (player.position.x, player.position.y + 10), 2)

    pygame.display.flip()


if __name__ == "__main__":
    main()
