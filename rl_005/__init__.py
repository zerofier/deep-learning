import os
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model, layers, optimizers

import common
import common.utilty


CURRENT_DIR = os.path.dirname(__file__)


class ReplayBuffer:
    def __init__(self, size) -> None:
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def add(self, observation, action, next_observation, reward, done):
        self.buffer.append((observation, action, next_observation, reward, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[i] for i in batch])
        return (
            tf.convert_to_tensor(observations, tf.float32),
            tf.convert_to_tensor(actions, tf.float32),
            tf.convert_to_tensor(next_observations, tf.float32),
            tf.convert_to_tensor(rewards, tf.float32)[:, tf.newaxis],
            tf.convert_to_tensor(dones, tf.float32)[:, tf.newaxis],
        )


class Critic(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d1 = layers.Dense(256, activation="relu")
        self.d2 = layers.Dense(256, activation="relu")
        self.output_q = layers.Dense(1)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.output_q(x)


class Actor(Model):
    def __init__(self, action_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_scale: np.ndarray = (action_spec.high - action_spec.low) / 2.0

        self.d1 = layers.Dense(256, activation="relu")
        self.d2 = layers.Dense(256, activation="relu")
        self.output_mean = layers.Dense(self.action_scale.shape[-1])
        self.output_log_stddev = layers.Dense(self.action_scale.shape[-1])

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        mean = self.output_mean(x)
        log_stddev = self.output_log_stddev(x)
        stddev = tf.exp(tf.clip_by_value(log_stddev, -20.0, 2.0))

        return mean, stddev

    def sample(self, observations):
        means, stddevs = self(observations)
        normal_dist = tfp.distributions.Normal(means, stddevs)
        zs = normal_dist.sample()
        actions = tf.tanh(zs) * self.action_scale
        log_pis = normal_dist.log_prob(zs) - tf.math.log(1 - tf.math.square(actions) + 1e-6)
        return actions, tf.reduce_mean(log_pis, axis=-1, keepdims=True), means


class SACAgent(common.Agent):
    def __init__(self, action_spec, gamma=0.99, tau=0.005) -> None:
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(action_spec)
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target1 = Critic()
        self.target2 = Critic()

        self.optimizer_actor = optimizers.Adam(3e-4)
        self.optimizer_critic1 = optimizers.Adam(3e-4)
        self.optimizer_critic2 = optimizers.Adam(3e-4)

        self.target_entropy = -np.prod(action_spec.shape)
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tf.exp(self.log_alpha)
        self.optimizer_alpha = optimizers.Adam(3e-4)

    def action_sample(self, observation):
        observations = tf.convert_to_tensor([observation], tf.float32)
        actions, _, _ = self.actor.sample(observations)
        return actions[0].numpy(), None

    def train(self, replay_buffer: ReplayBuffer, batch_size=0x100):
        observations, actions, next_observations, rewards, dones = replay_buffer.sample(batch_size)

        # calc target
        next_actions, next_log_pis, _ = self.actor.sample(next_observations)
        next_inputs = tf.concat([next_observations, next_actions], axis=-1)
        next_q1 = self.target1(next_inputs)
        next_q2 = self.target2(next_inputs)
        next_q = tf.minimum(next_q1, next_q2) - self.alpha * next_log_pis
        target_q = rewards + (1.0 - dones) * self.gamma * next_q

        inputs = tf.concat([observations, actions], axis=-1)
        # update critic1
        with tf.GradientTape() as tape1:
            q1 = self.critic1(inputs)
            critic1_loss = tf.reduce_mean(tf.math.square(q1 - target_q)) * 0.5
        critic1_grad = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        self.optimizer_critic1.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))

        # update critic2
        with tf.GradientTape() as tape2:
            q2 = self.critic2(inputs)
            critic2_loss = tf.reduce_mean(tf.math.square(q2 - target_q)) * 0.5
        critic2_grad = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.optimizer_critic2.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))

        # update actor
        with tf.GradientTape() as tape3:
            new_action, log_pis, _ = self.actor.sample(observations)
            new_inputs = tf.concat([observations, new_action], axis=-1)
            new_q1 = self.critic1(new_inputs)
            new_q2 = self.critic2(new_inputs)
            new_q = tf.minimum(new_q1, new_q2)
            actor_loss = tf.reduce_mean(self.alpha * log_pis - new_q)
        actor_grad = tape3.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # update alpha
        with tf.GradientTape() as tape4:
            alpha_loss = -tf.reduce_mean(self.log_alpha * (log_pis + self.target_entropy))
        alpha_grad = tape4.gradient(alpha_loss, [self.log_alpha])
        self.optimizer_alpha.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        self.alpha = np.exp(self.log_alpha)

        # update target
        for var, target_var in zip(self.critic1.variables, self.target1.variables):
            target_var.assign((1.0 - self.tau) * target_var + self.tau * var)

        for var, target_var in zip(self.critic2.variables, self.target2.variables):
            target_var.assign((1.0 - self.tau) * target_var + self.tau * var)

        return (actor_loss.numpy(), critic1_loss.numpy(), critic2_loss.numpy())

    def save(self, save_path):
        self.actor.save_weights(os.path.join(save_path, "actor.weights.h5"))
        self.critic1.save_weights(os.path.join(save_path, "critic1.weights.h5"))
        self.critic2.save_weights(os.path.join(save_path, "critic2.weights.h5"))
        self.target1.save_weights(os.path.join(save_path, "target1.weights.h5"))
        self.target2.save_weights(os.path.join(save_path, "target2.weights.h5"))

    def load(self, load_path):
        self.actor.load_weights(os.path.join(load_path, "actor.weights.h5"))
        self.critic1.load_weights(os.path.join(load_path, "critic1.weights.h5"))
        self.critic2.load_weights(os.path.join(load_path, "critic2.weights.h5"))
        self.target1.load_weights(os.path.join(load_path, "target1.weights.h5"))
        self.target2.load_weights(os.path.join(load_path, "target2.weights.h5"))


def training(episodes):
    env = gym.make("BipedalWalker-v3")
    agent = SACAgent(env.action_space)
    replay_buffer = ReplayBuffer(int(1e6))
    batch_size = 0x100
    exploration_count = 1600 * 6

    log_file = os.path.join(CURRENT_DIR, f"loss_history_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv")
    with open(log_file, "wt") as fd:
        fd.write("timestamp,loss,total_reward,total_steps\n")

    total_steps = 0
    for episode in range(episodes):
        observation, _ = env.reset()
        done = 0
        episode_steps = 0
        episode_reward = 0.0
        loss = (np.nan, np.nan, np.nan)
        while done == 0:
            if episode_steps + total_steps < exploration_count:
                action = env.action_space.sample()
            else:
                action, _ = agent.action_sample(observation)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)

            replay_buffer.add(observation, action, next_observation, reward, done)
            if len(replay_buffer) >= batch_size:
                loss = agent.train(replay_buffer, batch_size)

            episode_reward += reward
            observation = next_observation
            episode_steps += 1

        total_steps += episode_steps

        with open(log_file, "at") as fd:
            fd.write(f"{datetime.now()},\"{loss}\",{episode_reward},{episode_steps}\n")

        if (episode + 1) % 100 == 0:
            save_path = os.path.join(CURRENT_DIR, "models", datetime.now().strftime("%Y%m%dT%H%M%S"))
            os.makedirs(save_path, exist_ok=True)
            agent.save(save_path)
            common.utilty.record_video(gym.make("BipedalWalker-v3", render_mode="rgb_array"), agent, save_path)
