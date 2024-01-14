from typing import Tuple
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from gamelib import *
from dqn import DQN, Environment, ActionSpace

keras = tf.keras


# Parameters
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

I_SCREEN_HEIGHT = 1 / SCREEN_HEIGHT

FPS = 46

CAR_SCALING = 0.3

ACCELERATION = 72000
ANGULAR_ACCELERATION = 36000

AIR_RESISTANCE = 0.5
FRICTION_COEFFICIENT = 700
ANGULAR_FRICTION = 200
MAX_ANGULAR_VELOCITY = 400

SAFE_RADIUS = 400
TIME_TO_ESCAPE = 8

# 0 => Player controls criminal; ML agent controls police
# 1 => Player controls police; ML agent controls criminal
# 2 => ML Agents control both, but do not learn
# 3 => ML Agents control both, and criminal learns
# 4 => ML Agents control both, and police learns
ML_MODE = 4

C_DIST_REWARD = 5
C_CHASE_DURATION_REWARD = 2
P_CLOSE_REWARD = 5

EPISODES = 1

# Whether or not to render the game
RENDER = True


# Library objects
clock = pygame.time.Clock()


# Global variables
fps = FPS
spf = 1 / fps

screen: pygame.Surface
fullscreen: bool = False

criminal: Car = None
police: Car = None

timer: int


def main():
    global criminal, police, screen

    np.random.seed()
    pygame.init()

    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Car Simulation")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # fullscreen = False

    running = True

    setup()

    inputs = None


    if ML_MODE >= 3:
        # Setup DQN
        env = make_env()

        # setting up params
        lr = 0.001
        epsilon = 1.0
        epsilon_decay = 0.995
        gamma = 0.99
        training_episodes = EPISODES
        print('St')
        model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        model.train(env, training_episodes, False)

        # Save Everything
        save_dir = "saved_models"
        # Save trained model
        model.save(save_dir + "trained_model.keras")

        # Save Rewards list
        pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
        rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

        # plot reward in graph
        reward_df = pd.DataFrame(rewards_list)
        plot_df(reward_df, "Figure 1: Reward for each training episode", "Reward for each training episode", "Episode","Reward")

        # Test the model
        trained_model = keras.models.load_model(save_dir + "trained_model.h5")
        test_rewards = test_already_trained_model(trained_model)
        pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
        test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))

        plot_df2(pd.DataFrame(test_rewards), "Figure 2: Reward for each testing episode","Reward for each testing episode", "Episode", "Reward")
        print("Training and Testing Completed...!")

        # Run experiments for hyper-parameter
        run_experiment_for_lr()
        run_experiment_for_ed()
        run_experiment_for_gamma()
    else:
        inputs = InputHandler()


    reset()

    while running:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if update(input_update(inputs)) != 0:
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
    global criminal, police, timer

    while True:
        criminal.position = Vector(np.random.rand() * SCREEN_WIDTH, np.random.rand() * SCREEN_HEIGHT)
        police.position = Vector(np.random.rand() * SCREEN_WIDTH, np.random.rand() * SCREEN_HEIGHT)

        if criminal.position.sqrdistance(police.position) > (SAFE_RADIUS) * (SAFE_RADIUS):
            break
    
    criminal.rotation = np.random.rand() * 360
    police.rotation = np.random.rand() * 360

    criminal.rest()
    police.rest()

    timer = TIME_TO_ESCAPE * fps


def input_update(inputs: InputHandler) -> Tuple[Actions, Actions]:
    global screen, fullscreen

    inputs.update()
    inputs.get_global()

    if inputs.fullscreen_p:
        if fullscreen:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            fullscreen = False
        else:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
            fullscreen = True

    c_actions = inputs.get_actions() if ML_MODE == 0 else actions_from_nn(criminal, police)
    p_actions = inputs.get_actions() if ML_MODE == 1 else actions_from_nn(police, criminal)

    return (c_actions, p_actions)


def actions_from_nn(car: Car, other: Car) -> Actions:
    actions = Actions()
    actions.forward = False
    actions.backward = False
    actions.left = False
    actions.right = False
    return actions


# Returns 0 if game is still running, 1 if police wins, -1 if criminal wins
def update(actions: Tuple[Actions, Actions]) -> int:
    global criminal, police, timer

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
    
    timer -= 1
    
    if (
        timer <= 0
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
        0, # distance to opponent
        0, # opponent x
        0, # opponent y
        0, # opponent velocity magnitude
        0, # opponent velocity angle
        0, # opponent angle
        0, # opponent angular velocity
    ])

    env.reset = env_reset.__get__(env)
    env.step  = env_step.__get__(env)

    if RENDER:
        env.render = env_render.__get__(env)
    else:
        env.render = env_not_render.__get__(env)

    if ML_MODE == 3:
        env.reward_continuous = (lambda self: criminal.position.distance(police.position) * I_SCREEN_HEIGHT * C_DIST_REWARD * spf).__get__(env)
        env.reward_caught = (lambda self: -100 + timer * spf * C_CHASE_DURATION_REWARD).__get__(env)
        env.reward_escaped = (lambda self: 100).__get__(env)
    elif ML_MODE == 4:
        env.reward_continuous = (lambda self: P_CLOSE_REWARD * (1 - police.position.distance(criminal.position) * I_SCREEN_HEIGHT) * spf).__get__(env)
        env.reward_caught = (lambda self: 100 + self.reward_continuous() * (1 + TIME_TO_ESCAPE - timer)).__get__(env)
        env.reward_escaped = (lambda self: -100).__get__(env)

    return env


def get_state() -> np.ndarray:
    if ML_MODE == 3:
        return np.array([
            criminal.position.x, criminal.position.y,
            criminal.velocity.magnitude(), criminal.velocity.angle(),
            criminal.rotation, criminal.angular_velocity,
            criminal.position.distance(police.position),
            police.position.x, police.position.y,
            police.velocity.magnitude(), police.velocity.angle(),
            police.rotation, police.angular_velocity,
        ])
    elif ML_MODE == 4:
        return np.array([
            police.position.x, police.position.y,
            police.velocity.magnitude(), police.velocity.angle(),
            police.rotation, police.angular_velocity,
            police.position.distance(criminal.position),
            criminal.position.x, criminal.position.y,
            criminal.velocity.magnitude(), criminal.velocity.angle(),
            criminal.rotation, criminal.angular_velocity,
        ])


# Returns state
def env_reset(self: Environment) -> np.ndarray:
    reset()
    return get_state()


# Returns next_state, reward, done, info
def env_step(self: Environment, action: int) -> Tuple[np.ndarray, float, bool, None]:
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

    if not done:
        reward = self.reward_continuous()
    elif result == 1:
        reward = self.reward_caught()
    elif result == -1:
        reward = self.reward_escaped()

    return next_state, reward, done, None


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


def plot_experiments(df, chart_name, title, x_axis_label, y_axis_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8), title=title)
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim(y_limit)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def run_experiment_for_gamma():
    print('Running Experiment for gamma...')
    env = make_env()

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.9, 0.8, 0.7]
    training_episodes = EPISODES

    rewards_list_for_gammas = []
    for gamma_value in gamma_list:
        # save_dir = "hp_gamma_"+ str(gamma_value) + "_"
        model = DQN(env, lr, gamma_value, epsilon, epsilon_decay)
        print("Training model for Gamma: {}".format(gamma_value))
        model.train(training_episodes, False)
        rewards_list_for_gammas.append(model.rewards_list)

    pickle.dump(rewards_list_for_gammas, open("rewards_list_for_gammas.p", "wb"))
    rewards_list_for_gammas = pickle.load(open("rewards_list_for_gammas.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(gamma_list)):
        col_name = "gamma=" + str(gamma_list[i])
        gamma_rewards_pd[col_name] = rewards_list_for_gammas[i]
    plot_experiments(gamma_rewards_pd, "Figure 4: Rewards per episode for different gamma values",
                     "Figure 4: Rewards per episode for different gamma values", "Episodes", "Reward", (-600, 300))


def run_experiment_for_lr():
    print('Running Experiment for learning rate...')
    env = make_env()

    # setting up params
    lr_values = [0.0001, 0.001, 0.01, 0.1]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = EPISODES
    rewards_list_for_lrs = []
    for lr_value in lr_values:
        model = DQN(env, lr_value, gamma, epsilon, epsilon_decay)
        print("Training model for LR: {}".format(lr_value))
        model.train(training_episodes, False)
        rewards_list_for_lrs.append(model.rewards_list)

    pickle.dump(rewards_list_for_lrs, open("rewards_list_for_lrs.p", "wb"))
    rewards_list_for_lrs = pickle.load(open("rewards_list_for_lrs.p", "rb"))

    lr_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(lr_values)):
        col_name = "lr="+ str(lr_values[i])
        lr_rewards_pd[col_name] = rewards_list_for_lrs[i]
    plot_experiments(lr_rewards_pd, "Figure 3: Rewards per episode for different learning rates", "Figure 3: Rewards per episode for different learning rates", "Episodes", "Reward", (-2000, 300))


def run_experiment_for_ed():
    print('Running Experiment for epsilon decay...')
    env = make_env()

    # setting up params
    lr = 0.001
    epsilon = 1.0
    ed_values = [0.999, 0.995, 0.990, 0.9]
    gamma = 0.99
    training_episodes = EPISODES

    rewards_list_for_ed = []
    for ed in ed_values:
        save_dir = "hp_ed_"+ str(ed) + "_"
        model = DQN(env, lr, gamma, epsilon, ed)
        print("Training model for ED: {}".format(ed))
        model.train(training_episodes, False)
        rewards_list_for_ed.append(model.rewards_list)

    pickle.dump(rewards_list_for_ed, open("rewards_list_for_ed.p", "wb"))
    rewards_list_for_ed = pickle.load(open("rewards_list_for_ed.p", "rb"))

    ed_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes+1)))
    for i in range(len(ed_values)):
        col_name = "epsilon_decay = "+ str(ed_values[i])
        ed_rewards_pd[col_name] = rewards_list_for_ed[i]
    plot_experiments(ed_rewards_pd, "Figure 5: Rewards per episode for different epsilon(ε) decay", "Figure 5: Rewards per episode for different epsilon(ε) decay values", "Episodes", "Reward", (-600, 300))


if __name__ == "__main__":
    main()
