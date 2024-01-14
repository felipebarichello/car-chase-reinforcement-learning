import pygame
import numpy as np
from typing import Self


# Multiply by this to convert degrees to radians
DEG2RAD = np.pi / 180

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


class Actions:
    forward: bool
    backward: bool
    left: bool
    right: bool


class InputHandler:
    def __init__(self):
        self.fullscreen = False

    def update(self):
        self.pressed = pygame.key.get_pressed()

    def get_global(self):
        pressed = self.pressed
        last_fullscreen = self.fullscreen
        self.fullscreen = pressed[pygame.K_F11]
        self.fullscreen_p = self.fullscreen and not last_fullscreen
        return self

    def get_actions(self) -> Actions:
        pressed = self.pressed
        actions = Actions()
        actions.forward  = pressed[pygame.K_UP]    or pressed[ord("w")]
        actions.backward = pressed[pygame.K_DOWN]  or pressed[ord("s")]
        actions.left     = pressed[pygame.K_LEFT]  or pressed[ord("a")]
        actions.right    = pressed[pygame.K_RIGHT] or pressed[ord("d")]
        return actions


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

        friction_force: Vector = -self.velocity.normalized() * (self.friction_coefficient * np.fabs(np.sin(self.velocity.angle() - self.rotation * DEG2RAD)))
        delta_v: Vector = friction_force * spf

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

    def blit(self, screen):
        rotated_sprite = pygame.transform.rotate(self.sprite, self.rotation)

        angle: float = self.rotation * DEG2RAD
        radius: float = self.sprite.get_width() * 0.28
        center: Vector = Vector(self.position.x - rotated_sprite.get_width() * 0.5, self.position.y - rotated_sprite.get_height() * 0.5)
        final: Vector = Vector(center.x + radius * np.cos(angle), center.y - radius * np.sin(angle))

        screen.blit(rotated_sprite, (final.x, final.y))
