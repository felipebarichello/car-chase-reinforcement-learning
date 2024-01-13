import pygame
import numpy as np
from typing import Self

clock = pygame.time.Clock()

# Constants
DEG2RAD = np.pi / 180
FPS = 60
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
ACCELERATION = 1200
ANGULAR_ACCELERATION = 600
AIR_RESISTANCE = 0.5
FRICTION_COEFFICIENT = 500
ANGULAR_FRICTION = 200
MAX_ANGULAR_VELOCITY = 400


class Vector:
    def __init__(self, x, y):
        self.x: float = x
        self.y: float = y

    def __neg__(self) -> Self:
        return Vector(-self.x, -self.y)

    def __add__(self, other: Self) -> Self:
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Self:
        return Vector(scalar * self.x, scalar * self.y)

    def __truediv__(self, scalar: float) -> Self:
        if scalar == 0:
            return Vector(0, 0)

        return self * (1/scalar)

    def __pow__(self, power, modulo=None) -> Self:
        return Vector(np.sign(self.x) * np.fabs(self.x) ** power, np.sign(self.y) * np.fabs(self.y) ** power)

    def __str__(self) -> str:
        return "(%f, %f)" % (self.x, self.y)

    def magnitude(self) -> float:
        return np.sqrt(self.sqrmagnitude())

    def sqrmagnitude(self) -> float:
        return self.x * self.x + self.y * self.y

    def angle(self) -> float:
        if self.x == 0:
            return 0

        return np.arctan(self.y / self.x)

    def normalized(self) -> Self:
        magnitude: float = self.magnitude()

        if magnitude == 0:
            return Vector(1, 0)

        return self / self.magnitude()


class Car:
    def __init__(self, spawnpoint: Vector, sprite, acceleration: float, angular_acceleration: float, air_resistance: float, friction_coefficient: float, angular_friction: float):
        self.sprite = sprite

        self.position: Vector = spawnpoint
        self.rotation: float = 0

        self.acceleration: float = acceleration
        self.angular_acceleration: float = angular_acceleration

        self.velocity: Vector = Vector(0, 0)
        self.angular_velocity: float = 0

        self.air_resistance: float = air_resistance
        self.friction_coefficient: float = friction_coefficient
        self.angular_friction: float = angular_friction

        self.wheels: float = 0

    def accelerate(self, multiplier):
        angle = self.rotation * DEG2RAD
        self.velocity.x += multiplier * self.acceleration * np.cos(angle)
        self.velocity.y -= multiplier * self.acceleration * np.sin(angle)

    def rotate(self, multiplier):
        self.angular_velocity += multiplier * self.angular_acceleration

        if np.fabs(self.angular_velocity) > MAX_ANGULAR_VELOCITY:
            self.angular_velocity = MAX_ANGULAR_VELOCITY * np.sign(self.angular_velocity)

    def update(self, spf):
        self.position += self.velocity * spf
        self.rotation += self.angular_velocity * spf

        friction_force = -self.velocity.normalized() * (self.friction_coefficient * np.fabs(np.sin(self.velocity.angle() - self.rotation * DEG2RAD)))
        delta_v = friction_force * spf

        if self.velocity.magnitude() < delta_v.magnitude():
            self.velocity = Vector(0, 0)
        else:
            self.velocity += delta_v

        air_resistance_force = self.velocity * self.air_resistance
        self.velocity -= air_resistance_force * spf

        angular_friction_torque = -np.sign(self.angular_velocity) * self.angular_friction
        delta_rot = angular_friction_torque * spf

        if np.fabs(self.angular_velocity) < np.fabs(delta_rot):
            self.angular_velocity = 0
        else:
            self.angular_velocity += delta_rot


# Global variables
fps = FPS
spf = 1 / fps

player: Car = None


def main():
    pygame.init()

    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Car Simulation")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True

    player_sprite = pygame.image.load("red_car.png")
    player_sprite = pygame.transform.scale(player_sprite, (0.5 * player_sprite.get_width(), 0.5 * player_sprite.get_height()))

    spawnpoint = Vector(100, 100)
    player = Car(spawnpoint, player_sprite, ACCELERATION * spf, ANGULAR_ACCELERATION * spf, AIR_RESISTANCE, FRICTION_COEFFICIENT, ANGULAR_FRICTION)


    while running:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pressed = pygame.key.get_pressed()

        down = pressed[pygame.K_DOWN] or pressed[ord("s")]
        up = pressed[pygame.K_UP] or pressed[ord("w")]
        left = pressed[pygame.K_LEFT] or pressed[ord("a")]
        right = pressed[pygame.K_RIGHT] or pressed[ord("d")]

        # Handle input
        if down:
            player.accelerate(-1)
        if up:
            player.accelerate(1)
        if left:
            player.rotate(1)
        if right:
            player.rotate(-1)

        # Update transform
        player.update(spf)

        draw(screen, player)


def draw(screen, player: Car):
    screen.fill((100, 100, 100))
    rotated_player = pygame.transform.rotate(player.sprite, player.rotation)

    angle: float = player.rotation * DEG2RAD
    radius: float = player.sprite.get_width() / 2
    center: Vector = Vector(player.position.x - rotated_player.get_width() / 2, player.position.y - rotated_player.get_height() / 2)
    final: Vector = Vector(center.x + radius * np.cos(angle), center.y - radius * np.sin(angle))

    screen.blit(rotated_player, (final.x, final.y))

    pygame.display.flip()


if __name__ == "__main__":
    main()
