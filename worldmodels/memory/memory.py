import math
import os

import numpy as np
import tensorflow as tf

# https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py


def get_pi_idx(pis, threshold):
    """ Samples the probabilities of each mixture """
    if threshold is None:
        threshold = np.random.rand(1)

    pdf = 0.0
    #  one sample, one timestep
    for idx, prob in enumerate(pis):
        pdf += prob
        if pdf > threshold:
            return idx

    #  if we get to this point, something is wrong!
    print('pdf {} thresh {}'.format(pdf, threshold))
    return idx


class MLP(tf.keras.Model):
    """ used for testing only """
    def __init__(self, num_mix, hidden_nodes):
        super().__init__()
        self.perceptron = tf.keras.Sequential(
            [tf.keras.layers.Dense(
                24,
                dtype='float32',
                activation='tanh',
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.5)
            ),
             tf.keras.layers.Dense(num_mix * 3, dtype='float32')
            ]
        )

    def __call__(self, input_tensor):
        return self.perceptron(input_tensor)


class LSTM():
    """ car racing defaults """
    def __init__(
            self,
            input_dim,
            output_dim,
            num_timesteps,
            batch_size,
            nodes
    ):
        self.input_dim = input_dim
        self.nodes = nodes
        self.batch_size = batch_size

        input_layer = tf.keras.Input(shape=(num_timesteps, input_dim), batch_size=batch_size)

        cell = tf.keras.layers.LSTMCell(
            nodes,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
        )

        self.lstm = tf.keras.layers.RNN(
            cell,
            return_state=True,
            return_sequences=True,
            stateful=False
        )

        lstm_out, hidden_state, cell_state = self.lstm(input_layer)
        output = tf.keras.layers.Dense(output_dim)(lstm_out)

        self.net = tf.keras.Model(inputs=input_layer, outputs=[output, hidden_state, cell_state])

    def get_zero_hidden_state(self, inputs):
        #  inputs dont matter here - but batch size does!
        return [
            tf.zeros((inputs.shape[0], self.nodes)),
            tf.zeros((inputs.shape[0], self.nodes))
        ]

    def get_initial_state(self, inputs):
        return self.initial_state

    def __call__(self, inputs, state):
        self.initial_state = state
        self.lstm.get_initial_state = self.get_initial_state
        return self.net(inputs)


class GaussianMixture(tf.keras.Model):
    def __init__(self, num_features, num_mix, num_timesteps, batch_size):
        self.num_mix = num_mix

        #  (batch_size, num_timesteps, output_dim * num_mix * 3)
        #  3 = one pi, mu, sigma for each mixture
        mixture_dim = num_features * num_mix * 3

        input_layer = tf.keras.Input(shape=(num_timesteps, mixture_dim), batch_size=batch_size)

        #  (batch, time, num_features, num_mix * 3)
        expand = tf.reshape(input_layer, (-1, num_timesteps, num_features, num_mix * 3))

        #  (batch, time, num_features, num_mix)
        pi, mu, sigma = tf.split(expand, 3, axis=3)

        #  softmax the pi's (alpha in Bishop 1994)
        pi = tf.exp(tf.subtract(tf.reduce_max(pi, 3, keepdims=True), pi))
        pi = tf.divide(pi, tf.reduce_sum(pi, 3, keepdims=True))

        sigma = tf.maximum(sigma, 1e-8)
        sigma = tf.exp(sigma)

        super().__init__(inputs=[input_layer], outputs=[pi, mu, sigma])

    def kernel_probs(self, mu, sigma, next_latent):
        constant = 1 / math.sqrt(2 * math.pi)

        #  mu.shape
        #  (batch_size, num_timesteps, num_features, num_mix)

        #  next_latent.shape
        #  (batch_size, num_timesteps, num_features)
        #  -> (batch_size, num_timesteps, num_features, num_mix)
        next_latent = tf.expand_dims(next_latent, axis=-1)
        next_latent = tf.tile(next_latent, (1, 1, 1, self.num_mix))

        gaussian_kernel = tf.subtract(next_latent, mu)
        gaussian_kernel = tf.square(tf.divide(gaussian_kernel, sigma))
        gaussian_kernel = - 1/2 * gaussian_kernel
        conditional_probabilities = tf.divide(tf.exp(gaussian_kernel), sigma) * constant

        #  (batch_size, num_timesteps, num_features, num_mix)
        return conditional_probabilities

    def get_loss(self, mixture, next_latent):
        pi, mu, sigma = self(mixture)
        probs = self.kernel_probs(mu, sigma, next_latent)
        loss = tf.multiply(probs, pi)

        #  reduce along the mixes
        loss = tf.reduce_sum(loss, 3, keepdims=True)
        loss = -tf.math.log(loss)
        loss = tf.reduce_mean(loss)
        return loss


class Memory:
    """ initializes LSTM and Mixture models """
    def __init__(
            self,
            input_dim=35,
            output_dim=32,
            num_timesteps=999,
            batch_size=100,
            lstm_nodes=256,
            num_mix=5,
            grad_clip=1.0,
            initial_learning_rate=0.001,
            end_learning_rate=0.00001,
            epochs=1,
            batch_per_epoch=1,
            load_model=False,
            results_dir=None
    ):
        decay_steps = epochs * batch_per_epoch
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        mixture_dim = output_dim * num_mix * 3

        self.lstm = LSTM(
            input_dim,
            mixture_dim,
            num_timesteps,
            batch_size,
            lstm_nodes
        )

        self.mixture = GaussianMixture(
            output_dim,
            num_mix,
            num_timesteps,
            batch_size
        )

        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipvalue=grad_clip)

        self.models = {
            'lstm': self.lstm.net,
            'gaussian-mix': self.mixture
        }

        if load_model:
            self.load(results_dir)

    def save(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath, 'models')
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        for name, model in self.models.items():
            model.save_weights('{}/{}.h5'.format(filepath, name))

    def load(self, filepath):
        """ only model weights """
        filepath = os.path.join(filepath, 'models')
        print('loading model from {}'.format(filepath))

        for name, model in self.models.items():
            model.load_weights('{}/{}.h5'.format(filepath, name))
            self.models[name] = model

    def __call__(self, x, state, temperature, threshold=None):
        """
        forward pass

        hardcoded for a single step - because we want to pass state
        inbetween
        """
        x = tf.reshape(x, (1, 1, self.input_dim))
        assert x.shape[0] == 1

        temperature = np.array(temperature).reshape(1, 1)
        assert temperature.shape[0] == x.shape[0]

        mixture, h_state, c_state = self.lstm(x, state)

        pi, mu, sigma = self.mixture(mixture) #, temperature

        #  single sample, single timtestep
        pi = np.array(pi).reshape(self.output_dim, pi.shape[3])
        mu = np.array(mu).reshape(self.output_dim, mu.shape[3])
        sigma = np.array(sigma).reshape(self.output_dim, sigma.shape[3])

        #  reset every forward pass
        idxs = np.zeros(self.output_dim)
        mus = np.zeros(self.output_dim)
        sigmas = np.zeros(self.output_dim)
        y = np.zeros(self.output_dim)

        for num in range(self.output_dim):
            idx = get_pi_idx(pi[num, :], threshold=threshold)

            idxs[num] = idx
            mus[num] = mu[num, idx]
            sigmas[num] = sigma[num, idx]

            y[num] = mus[num] + np.random.randn() * sigmas[num] * np.sqrt(temperature)

        #  check no zeros in pis
        assert sum(idxs) > 0

        return y, h_state, c_state

    def train_op(self, x, y, state):
        """ backward pass """
        with tf.GradientTape() as tape:
            out, _, _ = self.lstm(x, state)
            loss = self.mixture.get_loss(out, y)
            gradients = tape.gradient(loss, self.lstm.net.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.lstm.net.trainable_variables)
        )
        return loss
