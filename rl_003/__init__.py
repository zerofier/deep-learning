"""REINFORCE Agent to """

import gc
import os
from datetime import datetime

import keras
import gymnasium as gym
import numpy as np
import tensorflow as tf

import common
import common.utilty


CURRENT_DIR = os.path.dirname(__file__)

EPS = 1e-6
LOG_2PI = np.log(2.0 * np.pi)

LOG_FILE = os.path.join(CURRENT_DIR, f"loss_history_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv")


class StddevLimter(keras.Layer):
    """Limited for stddev value from NN output

    See:
        https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    """

    def call(self, inputs):
        return tf.math.log(tf.exp(inputs) + 1.0) + EPS


class REINFORCE(common.Agent):
    def __init__(self, observation_shape, action_shape, model_path=None) -> None:
        self.gamma = 0.99

        self.observation_shape = observation_shape
        self.action_shape = action_shape

        inputs = keras.Input(self.observation_shape)
        hidden_layer1 = keras.layers.Dense(256, keras.activations.relu, name="hidden_layer1")(inputs)
        hidden_layer2 = keras.layers.Dense(256, keras.activations.relu, name="hidden_layer2")(hidden_layer1)
        mean_output = keras.layers.Dense(self.action_shape[0], keras.activations.linear, name="mean_output")(
            hidden_layer2
        )
        stddev_hidden = keras.layers.Dense(self.action_shape[0], keras.activations.linear, name="stddev_hidden")(
            hidden_layer2
        )
        stddev_output = StddevLimter(name="stddev_output")(stddev_hidden)

        self.pi = keras.Model(inputs=inputs, outputs=[mean_output, stddev_output])
        self.optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        if model_path:
            self.pi = keras.models.load_model(os.path.join(model_path, "model.keras"))

    def action_sample(self, observation):
        mean, stddev = self.pi(tf.reshape(observation, (-1, self.observation_shape[-1])), training=False)
        raw_action = tf.random.normal(tf.shape(mean), mean, stddev)
        action = tf.tanh(raw_action)
        return action[0], raw_action[0]


def training(episodes, model_path=None):
    env = gym.make("BipedalWalker-v3")

    agent = REINFORCE(env.observation_space.shape, env.action_space.shape, model_path)
    agent.pi.summary()

    with open(LOG_FILE, "wt") as fd:
        fd.write("timestamp,loss,total_reward,total_steps\n")

    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        observ_history = []
        action_history = []
        reward_history = []

        while not done:
            action, raw_action = agent.action_sample(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)

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
        gs = tf.constant(g_history, dtype=tf.float32)
        # base line
        gs -= tf.reduce_mean(gs)

        with tf.GradientTape() as tape:
            means, stddevs = agent.pi(observations, training=True)
            # log_prob = (tf.reduce_sum(tf.square((actions - means) / stddevs), axis=1) + LOG_2PI) * -0.5 - tf.reduce_sum(
            #     tf.math.log(stddevs), axis=1
            # )
            log_prob = tf.reduce_sum(
                (tf.square((actions - means) / stddevs) + LOG_2PI) * -0.5 - tf.math.log(stddevs), axis=1
            )
            log_prob -= tf.reduce_sum(tf.math.log(1.0 - tf.square(tf.tanh(actions)) + EPS), axis=1)
            loss = -tf.reduce_sum(log_prob * gs)

        grads = tape.gradient(loss, agent.pi.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.pi.trainable_variables))

        total_reward = np.sum(reward_history)

        with open(LOG_FILE, "at") as fd:
            fd.write(f"{datetime.now()},{loss},{total_reward},{len(reward_history)}\n")

        if (episode + 1) % 1000 == 0:
            save_dir = os.path.join(CURRENT_DIR, "model", datetime.now().strftime("%Y%m%dT%H%M%S"))
            os.makedirs(save_dir, exist_ok=True)
            agent.pi.save(os.path.join(save_dir, "model.keras"))
            common.utilty.record_video(gym.make("BipedalWalker-v3", render_mode="rgb_array"), agent, save_dir)

        gc.collect()
