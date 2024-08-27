"""REINFORCE Agent to """

import gc
import os
from datetime import datetime

import keras
import gymnasium as gym
import numpy as np
import tensorflow as tf


CURRENT_DIR = os.path.dirname(__file__)

EPS = 1e-6
LOG_2PI = np.log(2.0 * np.pi)


class StddevLimter(keras.Layer):
    """Limited for stddev value from NN output

    See:
        https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    """

    def call(self, inputs):
        return tf.math.log(tf.exp(inputs) + 1.0) + EPS


class REINFORCE:
    def __init__(self, observation_shape, action_shape, model_path=None) -> None:
        self.gamma = 0.99

        inputs = keras.Input(observation_shape)
        hidden_layer1 = keras.layers.Dense(256, keras.activations.relu, name="hidden_layer1")(inputs)
        hidden_layer2 = keras.layers.Dense(256, keras.activations.relu, name="hidden_layer2")(hidden_layer1)
        mean_output = keras.layers.Dense(action_shape[0], keras.activations.linear, name="mean_output")(hidden_layer2)
        stddev_hidden = keras.layers.Dense(action_shape[0], keras.activations.linear, name="stddev_hidden")(
            hidden_layer2
        )
        stddev_output = StddevLimter(name="stddev_output")(stddev_hidden)

        self.pi = keras.Model(inputs=inputs, outputs=[mean_output, stddev_output])
        self.optimizer = keras.optimizers.Adam(learning_rate=3e-4)

    def action_sample(self, observation):
        mean, stddev = self.pi(tf.reshape(observation, (-1, self.pi.input_shape[-1])), training=False)
        raw_action = tf.random.normal(tf.shape(mean), mean, stddev)
        raw_action = raw_action
        action = tf.tanh(raw_action)
        return action, raw_action


def training():
    env = gym.make("BipedalWalker-v3")

    agent = REINFORCE(env.observation_space.shape, env.action_space.shape)
    agent.pi.summary()

    with open("loss_history.csv", "wt") as fd:
        fd.write("timestamp,loss,total_reward,total_steps\n")

    for episode in range(10000):
        observation, _ = env.reset()
        done = False
        observ_history = []
        action_history = []
        reward_history = []

        while not done:
            action, raw_action = agent.action_sample(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action[0])

            observ_history.append(observation)
            action_history.append(raw_action)
            reward_history.append(reward)

            observation = next_observation
            done = terminated or truncated

        observations = tf.constant(observ_history)
        actions = tf.concat(action_history, axis=0)

        g_history = []
        g = 0
        for reward in reward_history[::-1]:
            g = reward + agent.gamma * g
            g_history.insert(0, g)
        gs = tf.constant(g_history, dtype=tf.float32, shape=(len(g_history), 1))

        with tf.GradientTape() as tape:
            means, stddevs = agent.pi(observations, training=True)
            log_prob = tf.reduce_sum(
                (LOG_2PI + tf.square((actions - means) / stddevs)) * -0.5 - tf.math.log(stddevs), axis=1, keepdims=True
            )
            log_prob -= tf.reduce_sum(tf.math.log(1.0 - tf.square(tf.tanh(actions)) + EPS), axis=1, keepdims=True)
            loss = tf.reduce_sum(log_prob * gs * -1.0)

        grads = tape.gradient(loss, agent.pi.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.pi.trainable_variables))

        total_reward = np.sum(reward_history)

        with open("loss_history.csv", "at") as fd:
            fd.write(f"{datetime.now()},{loss},{total_reward},{len(reward_history)}\n")

        if (episode + 1) % 1000 == 0:
            save_dir = os.path.join(CURRENT_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))
            os.makedirs(save_dir, exist_ok=True)
            agent.pi.save(os.path.join(save_dir, "model.keras"))

        gc.collect()
