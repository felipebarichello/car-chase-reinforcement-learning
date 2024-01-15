from typing import Tuple
import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf


keras = tf.keras


class ActionSpace:
    def __init__(self, n: int):
        self.n = n


class Environment:
    action_space: ActionSpace
    observation_space: np.ndarray
    quit_requested: bool

    def __init__(self):
        self.quit_requested = False

    def render(self) -> None:
        pass

    # Returns state
    def reset(self) -> np.ndarray:
        pass

    # Returns next_state, reward, done, info
    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, None]:
        pass

    # Returns next_state, rewards, done, info
    def multiagent_step(self, actions: np.array) -> Tuple[np.ndarray, np.array, bool, None]:
        pass


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        self.model = self.initialize_model()

    def initialize_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(512, input_dim=self.num_observation_space, activation=keras.activations.relu))
        model.add(keras.layers.Dense(256, activation=keras.activations.relu))
        model.add(keras.layers.Dense(self.num_action_space, activation=keras.activations.linear))

        # Compile the model
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr))
        # print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def train(self, env: Environment, num_episodes=2000, can_stop=False, num_steps=1000):
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            state = np.reshape(state, [1, self.num_observation_space])
            for step in range(num_steps):
                env.render()
                received_action = self.get_action(state)
                next_state, reward, done, info = env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                # Store the experience in replay memory
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_reply()

                if done:
                    break
            
            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 200 and can_stop:
                print("DQN Training Complete...")
                break
            print(episode, "\t: Episode || Reward: ",reward_for_episode, "\t|| Average Reward: ",last_rewards_mean, "\t epsilon: ", self.epsilon )

            if env.quit_requested:
                break

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)

    def train_multiagent(networks, env: Environment, num_episodes=2000, can_stop=False, num_steps=1000):
        for episode in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, networks[0].num_observation_space])

            for net in networks:
                net._reward_for_episode = 0

            for step in range(num_steps):
                env.render()
                received_actions = []

                for net in networks:
                    received_actions.append(net.get_action(state))

                next_state, rewards, done, info = env.multiagent_step(received_actions)

                next_state = np.reshape(next_state, [1, networks[0].num_observation_space])

                # Store the experience in replay memory
                for i, net in enumerate(networks):
                    net.add_to_replay_memory(state, received_actions[i], rewards[i], next_state, done)
                    net._reward_for_episode += rewards[i]
                    net.update_counter()
                    net.learn_and_update_weights_by_reply()

                state = next_state

                if done:
                    break
            
            for i, net in enumerate(networks):
                net.rewards_list.append(net._reward_for_episode)

                # Decay the epsilon after each experience completion
                if net.epsilon > net.epsilon_min:
                    net.epsilon *= net.epsilon_decay

                # Check for breaking condition
                net._last_rewards_mean = np.mean(net.rewards_list[-100:])
                if net._last_rewards_mean > 200 and can_stop:
                    print("DQN Training Complete...")
                    break

                print("Net ", i, ": ", episode, "\t: Episode || Reward: ",net._reward_for_episode, "\t|| Average Reward: ",net._last_rewards_mean, "\t epsilon: ", net.epsilon )

            if env.quit_requested:
                break
