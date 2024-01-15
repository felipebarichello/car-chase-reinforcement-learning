from typing import Tuple
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os

from gamelib import *
from dqn import DQN, Environment, ActionSpace

keras = tf.keras


# Parameters
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

I_SCREEN_WIDTH = 1 / SCREEN_WIDTH

FPS = 60

CAR_SCALING = 0.5

ACCELERATION = 72000
ANGULAR_ACCELERATION = 36000

AIR_RESISTANCE = 0.5
FRICTION_COEFFICIENT = 700
ANGULAR_FRICTION = 200
MAX_ANGULAR_VELOCITY = 400

SAFE_RADIUS = 400
SECONDS_TO_ESCAPE = 8

# 0 => Player controls criminal; ML agent controls police
# 1 => Player controls police; ML agent controls criminal
# 2 => ML Agents control both, but do not learn
# 3 => ML Agents control both, and criminal learns
# 4 => ML Agents control both, and police learns
# 5 => ML Agents control both, and both learn
ML_MODE = 5

C_DIST_REWARD = 60
C_CHASE_DURATION_REWARD = 5
P_CLOSE_REWARD = 600

MAX_EPISODES = 10000

MODEL_DIR = "saved_models/"

# Whether or not to render the game when ML_MODE greater than 2
RENDER = True


# Library objects
clock = pygame.time.Clock()


# Global variables
fps = FPS
spf = 1 / fps

quit_requested: bool

inputs: InputHandler
screen: pygame.Surface
fullscreen: bool = False

criminal: Car = None
police: Car = None

frames_to_escape: int


def main():
    global criminal, police, screen, quit_requested, inputs

    np.random.seed()
    pygame.init()

    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Car Simulation")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # fullscreen = False

    quit_requested = False

    setup()

    inputs = InputHandler()


    if ML_MODE >= 3:
        # Setup DQN
        env = make_env()

        # setting up params
        lr = 0.001
        epsilon = 1.0
        epsilon_decay = 0.995
        gamma = 0.99
        training_episodes = MAX_EPISODES
        save_dir: any

        if ML_MODE == 3:
            save_dir = MODEL_DIR + "criminal/"
        elif ML_MODE == 4:
            save_dir = MODEL_DIR + "police/"
        elif ML_MODE == 5:
            save_dir = [MODEL_DIR + "criminal/", MODEL_DIR + "police/"]

        if ML_MODE == 5:
            model = [DQN(env, lr, gamma, epsilon, epsilon_decay), DQN(env, lr, gamma, epsilon, epsilon_decay)]
        else:
            model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        
        try:
            if ML_MODE == 5:
                model[0].model = keras.models.load_model(save_dir[0] + "model.keras")
                model[0].epsilon = pickle.loads(open(save_dir[0] + "network.pkl", "rb").read())
                model[1].model = keras.models.load_model(save_dir[1] + "model.keras")
                model[1].epsilon = pickle.loads(open(save_dir[1] + "network.pkl", "rb").read())
            else:
                model.model = keras.models.load_model(save_dir + "model.keras")
                model.epsilon = pickle.loads(open(save_dir + "network.pkl", "rb").read())
        except:
            pass

        if ML_MODE == 5:
            DQN.train_multiagent(model, env, training_episodes, False, SECONDS_TO_ESCAPE * fps * 2) # Should never be reached
        else:
            model.train(env, training_episodes, False, SECONDS_TO_ESCAPE * fps * 2)

        # Save trained model
        if ML_MODE == 5:
            for i in range(2):
                os.makedirs(save_dir[i], exist_ok=True)
                model[i].save(save_dir[i] + "model.keras")

                with open(save_dir[i] + "network.pkl", "wb") as file:
                    file.write(pickle.dumps(model[i].epsilon))

                # Save Rewards list
                pickle.dump(model[i].rewards_list, open(save_dir[i] + "rewards_list.p", "wb"))
                rewards_list = pickle.load(open(save_dir[i] + "rewards_list.p", "rb"))
        else:   
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir + "model.keras")

            with open(save_dir + "network.pkl", "wb") as file:
                file.write(pickle.dumps(model.epsilon))

            # Save Rewards list
            pickle.dump(model.rewards_list, open(save_dir + "rewards_list.p", "wb"))
            rewards_list = pickle.load(open(save_dir + "rewards_list.p", "rb"))

            # plot reward in graph
            reward_df = pd.DataFrame(rewards_list)
            plot_df(reward_df, "Figure 1: Reward for each training episode", "Reward for each training episode", "Episode","Reward")

        # # Test the model
        # trained_model = keras.models.load_model(save_dir + "trained_model.keras")
        # test_rewards = test_already_trained_model(trained_model)
        # pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
        # test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))

        # plot_df2(pd.DataFrame(test_rewards), "Figure 2: Reward for each testing episode","Reward for each testing episode", "Episode", "Reward")
        # print("Training and Testing Completed...!")

        # Run experiments for hyper-parameter
        # run_experiment_for_lr()
        # run_experiment_for_ed()
        # run_experiment_for_gamma()

        return
    

    if ML_MODE == 0 or ML_MODE == 2:
        # Criminal is ML Agent
        criminal.model = keras.models.load_model(MODEL_DIR + "criminal/model.keras")
        criminal.auto = lambda state: actions_from_net(criminal, state)

    if ML_MODE == 1 or ML_MODE == 2:
        # Police is ML Agent
        police.model = keras.models.load_model(MODEL_DIR + "police/model.keras")
        police.auto = lambda state: actions_from_net(police, state)


    reset()

    while True:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True

        if quit_requested:
            break

        if update(input_update()) != 0:
            reset()
            continue

        draw()


def setup():
    global criminal, police

    criminal_sprite = pygame.image.load("red_car.png")
    criminal_sprite = pygame.transform.scale(criminal_sprite, (CAR_SCALING * criminal_sprite.get_width(), CAR_SCALING * criminal_sprite.get_height()))
    criminal = Car(Vector(0, 0), criminal_sprite, ACCELERATION * spf, ANGULAR_ACCELERATION * spf, AIR_RESISTANCE, FRICTION_COEFFICIENT, ANGULAR_FRICTION)

    police_sprite   = pygame.image.load("police_car.png")
    police_sprite   = pygame.transform.scale(police_sprite, (CAR_SCALING * police_sprite.get_width(), CAR_SCALING * police_sprite.get_height()))
    police = Car(Vector(0, 0), police_sprite, ACCELERATION * spf, ANGULAR_ACCELERATION * spf, AIR_RESISTANCE, FRICTION_COEFFICIENT, ANGULAR_FRICTION)


def reset():
    global criminal, police, frames_to_escape

    while True:
        criminal.position = Vector(np.random.rand() * SCREEN_WIDTH * 0.8 + SCREEN_WIDTH * 0.1, np.random.rand() * SCREEN_HEIGHT * 0.7 + SCREEN_HEIGHT * 0.15)
        police.position   = Vector(np.random.rand() * SCREEN_WIDTH * 0.8 + SCREEN_WIDTH * 0.1, np.random.rand() * SCREEN_HEIGHT * 0.7 + SCREEN_HEIGHT * 0.15)

        if criminal.position.sqrdistance(police.position) > (SAFE_RADIUS) * (SAFE_RADIUS):
            break
    
    criminal.rotation = np.random.rand() * 360
    police.rotation = np.random.rand() * 360

    criminal.rest()
    police.rest()

    frames_to_escape = SECONDS_TO_ESCAPE * fps


def input_update() -> Tuple[Actions, Actions]:
    global screen, fullscreen, quit_requested, inputs

    inputs.update()
    inputs.get_global()

    if inputs.quit:
        quit_requested = True

    if inputs.fullscreen_p:
        if fullscreen:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            fullscreen = False
        else:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
            fullscreen = True
        
    state = get_state()

    c_actions = inputs.get_actions() if ML_MODE == 1 else criminal.auto(state)
    p_actions = inputs.get_actions() if ML_MODE == 0 else police.auto(state)

    return (c_actions, p_actions)


def actions_from_net(agent: Car, state: np.ndarray) -> Actions:
    state = np.reshape(state, [1, state.shape[0]])
    selected = np.argmax(agent.model.predict(state))

    actions = Actions()
    actions.forward  = True if selected <  3 else False
    actions.backward = True if selected >= 6 else False
    actions.left     = True if selected % 3 == 0 else False
    actions.right    = True if selected % 3 == 2 else False
    return actions


# Returns 0 if game is still running, 1 if police wins, -1 if criminal wins
def update(actions: Tuple[Actions, Actions]) -> int:
    global criminal, police, frames_to_escape

    # Handle actions
    criminal.act(actions[0], spf)
    police.act(actions[1], spf)

    # Update transform
    criminal.update(spf)
    police.update(spf)
    
    police_hitbox = police.get_hitbox()
    criminal_hitbox = criminal.get_hitbox()

    if (
        criminal_hitbox.colliderect(police_hitbox)
        or criminal_hitbox.left < 0 or criminal_hitbox.right > (SCREEN_WIDTH)
        or criminal_hitbox.top  < 0 or criminal_hitbox.bottom > (SCREEN_HEIGHT)
    ):
        return 1
    
    frames_to_escape -= 1
    
    if (
        frames_to_escape <= 0
        or police_hitbox.left < 0 or police_hitbox.right > (SCREEN_WIDTH)
        or police_hitbox.top  < 0 or police_hitbox.bottom > (SCREEN_HEIGHT)
    ):
        return -1
    
    return 0


def draw():
    global screen, criminal, police

    screen.fill((100, 100, 100))
    
    criminal.blit(screen)
    police.blit(screen)

    # Draw hitbox
    # pygame.draw.rect(screen, (0, 0, 255), criminal.get_hitbox(), 2)
    # pygame.draw.rect(screen, (0, 0, 255), police.get_hitbox(), 2)

    # Draws a cross on relevant points
    # pygame.draw.line(screen, (0, 255, 0), (player.position.x - 10, player.position.y), (player.position.x + 10, player.position.y), 2)
    # pygame.draw.line(screen, (0, 255, 0), (player.position.x, player.position.y - 10), (player.position.x, player.position.y + 10), 2)

    pygame.display.flip()


def make_env() -> Environment:
    env = Environment()
    env.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    env.action_space = ActionSpace(9) # forward, backward, left, right

    env.observation_space = np.array([
        0, # x position
        0, # y position
        0, # velocity magnitude
        0, # velocity angle
        0, # angle
        0, # angular velocity
        0, # angle to opponent
        0, # distance to opponent
        0, # angle to opponent
        0, # opponent x
        0, # opponent y
        0, # opponent velocity magnitude
        0, # opponent velocity angle
        0, # opponent angle
        0, # opponent angular velocity
    ])

    env.reset = env_reset.__get__(env)
    env.step  = env_step.__get__(env)
    env.multiagent_step = env_multiagent_step.__get__(env)

    if RENDER:
        env.render = env_render.__get__(env)
    else:
        env.render = env_not_render.__get__(env)

    env.reward_continuous = (lambda self, c: (
        criminal.position.distance(police.position) * I_SCREEN_WIDTH * C_DIST_REWARD * spf if c == 0 else (
        P_CLOSE_REWARD * (1 - police.position.distance(criminal.position) * I_SCREEN_WIDTH) * spf)
    )).__get__(env)

    env.reward_caught = (lambda self, c: (
        -100 + frames_to_escape * spf * C_CHASE_DURATION_REWARD if c == 0 else (
        100 + self.reward_continuous(c) * (SECONDS_TO_ESCAPE * fps - frames_to_escape))
    )).__get__(env)

    env.reward_escaped = (lambda self, c: 100 if c == 0 else -100).__get__(env)

    return env


def get_state() -> np.ndarray:
    angle_between = (criminal.position - police.position).angle()

    return np.array([
        criminal.position.x, criminal.position.y,
        criminal.velocity.magnitude(), criminal.velocity.angle(),
        normalize_angle(criminal.rotation), criminal.angular_velocity,
        normalize_angle(criminal.rotation) * DEG2RAD - angle_between,
        criminal.position.distance(police.position),
        normalize_angle(police.rotation) * DEG2RAD - angle_between,
        police.position.x, police.position.y,
        police.velocity.magnitude(), police.velocity.angle(),
        normalize_angle(police.rotation), police.angular_velocity,
    ])


# Returns state
def env_reset(self: Environment) -> np.ndarray:
    reset()
    return get_state()


# Returns next_state, reward, done, info
def env_step(self: Environment, action: int) -> Tuple[np.ndarray, float, bool, None]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.quit_requested = True

    inputs.update()
    inputs.get_global()

    if inputs.quit:
        self.quit_requested = True

    agent_actions = Actions()
    agent_actions.forward  = True if action <  3 else False
    agent_actions.backward = True if action >= 6 else False
    agent_actions.left     = True if action % 3 == 0 else False
    agent_actions.right    = True if action % 3 == 2 else False

    other_actions = Actions()
    other_actions.forward = 0
    other_actions.backward = 0
    other_actions.left = 0
    other_actions.right = 0

    result: int

    if ML_MODE == 3:
        result = update((agent_actions, other_actions))
    elif ML_MODE == 4:
        result = update((other_actions, agent_actions))
        
    next_state = get_state()
    reward = 0
    done = result != 0

    car_n = 0 if ML_MODE == 3 else 1

    if not done:
        reward = self.reward_continuous(car_n)
    elif result == 1:
        reward = self.reward_caught(car_n)
    elif result == -1:
        reward = self.reward_escaped(car_n)

    return next_state, reward, done, None


def env_multiagent_step(self: Environment, actions: np.array) -> Tuple[np.ndarray, np.array, bool, None]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.quit_requested = True

    inputs.update()
    inputs.get_global()

    if inputs.quit:
        self.quit_requested = True

    criminal_actions = Actions()
    criminal_actions.forward  = True if actions[0] <  3 else False
    criminal_actions.backward = True if actions[0] >= 6 else False
    criminal_actions.left     = True if actions[0] % 3 == 0 else False
    criminal_actions.right    = True if actions[0] % 3 == 2 else False

    police_actions = Actions()
    police_actions.forward  = True if actions[1] <  3 else False
    police_actions.backward = True if actions[1] >= 6 else False
    police_actions.left     = True if actions[1] % 3 == 0 else False
    police_actions.right    = True if actions[1] % 3 == 2 else False

    result: int

    result = update((criminal_actions, police_actions))
        
    next_state = get_state()
    done = result != 0
    rewards = [0, 0]

    if not done:
        rewards[0] = self.reward_continuous(0)
        rewards[1] = self.reward_continuous(1)
    elif result == 1:
        rewards[0] = self.reward_caught(0)
        rewards[1] = self.reward_caught(1)
    elif result == -1:
        rewards[0] = self.reward_escaped(0)
        rewards[1] = self.reward_escaped(1)

    return next_state, rewards, done, None


def env_render(env: Environment) -> None:
    draw()


def env_not_render(env: Environment) -> None:
    pass


def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = make_env()
    print("Starting Testing of the trained model...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward
            if done:
                break
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episode || Reward: ", reward_for_episode)

    return rewards_list


def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((-400, 300))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)


def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim((0, 300))
    plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig(chart_name)


if __name__ == "__main__":
    main()
