from typing import Tuple

import tensorflow as tf
import keras


EPS = 1e-6


class PolicyModel(keras.Model):
    def __init__(self, input_shape, output_size, **kwargs):
        super().__init__(**kwargs)
        self.inputs = keras.Input(input_shape)
        self.hidden_layers = [
            keras.layers.Dense(256, keras.activations.relu, name="share_1"),
            keras.layers.Dense(256, keras.activations.relu, name="share_2"),
        ]
        self.mean_output = keras.layers.Dense(output_size, keras.activations.linear, name="mean_out")
        self.stdd_output = keras.layers.Dense(output_size, keras.activations.linear, name="stdd_out")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)

        mean = self.mean_output(x)
        stdd = tf.math.log(1 + tf.exp(self.stdd_output(x)))

        return mean, stdd


class Agent:
    def __init__(self, obs_space_dim, act_space_dim) -> None:
        self.optmizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.gamma = 0.99
        self.pi = PolicyModel(obs_space_dim, act_space_dim)
        self.eps = 1e-6

    def action_sample(self, obs) -> Tuple[tf.Tensor, tf.Tensor]:
        outputs: Tuple[tf.Tensor, tf.Tensor] = self.pi(obs)
        action_raw = tf.random.normal(outputs[0].shape, outputs[0], outputs[1] + EPS)
        return action_raw, tf.tanh(action_raw)

    