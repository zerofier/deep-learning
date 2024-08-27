import gymnasium as gym
import numpy as np
import tensorflow as tf
# import keras

from tensorflow.python import keras

from common import Agent

env = gym.make("BipedalWalker-v3")
"""The enviroment of BipedalWalker-v3"""


class Model(tf.keras.Model, Agent):
    """The Model of Agent to enviroment"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stddev = 1.0

        self.__dense1 = tf.keras.layers.Dense(32, tf.nn.sigmoid)
        self.__dense2 = tf.keras.layers.Dense(16, tf.nn.sigmoid)
        self.__output = tf.keras.layers.Dense(env.action_space.shape[0])

        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs, training=None, mask=None):
        x = self.__dense1(inputs)
        x = self.__dense2(x)
        x = self.__output(x)

        return x, tf.tanh(x)

    def action_sample(self, observation):
        _, mean = self(observation)

        return tf.random.normal(tf.shape(mean), mean, self.stddev)


def training(model: keras.Model, experience: list[dict], epoch=10):

    print("observation", experience[0]["observation"])
    
    observations = np.array([__dict["observation"] for __dict in experience]) 
    actions = np.array([__dict["action"] for __dict in experience])
    rewards = np.array([__dict["reward"] for __dict in experience])
    next_observations = np.array([__dict["next_observation"] for __dict in experience]) 

    for _ in range(epoch):
        
        with tf.GradientTape() as tape:
            prod_raw, prod = model(observations, train=True)
            