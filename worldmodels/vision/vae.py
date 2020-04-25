import os

import tensorflow as tf


class VAE(tf.keras.Model):
    kl_tolerance = 0.5

    def __init__(
        self,
        latent_dim,
        results_dir,
        learning_rate=0.0001,
        load_model=True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        #  the encoder
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=4,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=4,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=4,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=4,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        #  the decoder
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=4*256, activation=tf.nn.relu),
            tf.keras.layers.Reshape([-1, 1, 4*256]),
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=5,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=6,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=6,
                strides=(2, 2),
                activation='sigmoid'
            )
        ])

        self.models = {
            'inference': self.inference_net,
            'generative': self.generative_net
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

    def forward(self, batch):
        """ images to reconstructed """
        means, logvars = self.encode(batch)
        latent = self.reparameterize(means, logvars)
        return self.decode(latent)

    def encode(self, batch):
        """ images to latent statistics """
        mean, logvar = tf.split(
            self.inference_net(batch), num_or_size_splits=2, axis=1
        )
        return mean, logvar

    def reparameterize(self, means, logvars):
        """ latent statistics to latent """
        epsilon = tf.random.normal(shape=means.shape)
        return means + epsilon * tf.exp(logvars * .5)

    def decode(self, latent):
        """ latent to reconstructed """
        return self.generative_net(latent)

    def loss(self, batch):
        """ batch to loss """
        means, logvars = self.encode(batch)
        latent = self.reparameterize(means, logvars)
        generated = self.decode(latent)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(batch - generated), axis=[1, 2, 3])
        )

        unclipped_kl_loss = - 0.5 * tf.reduce_sum(
            1 + logvars - tf.square(means) - tf.exp(logvars),
            axis=1
        )

        kl_loss = tf.reduce_mean(
            tf.maximum(unclipped_kl_loss, self.kl_tolerance * self.latent_dim)
        )
        return {
            'reconstruction-loss': reconstruction_loss,
            'unclipped-kl-loss': unclipped_kl_loss,
            'kl-loss': kl_loss
        }

    def backward(self, batch):
        """ images to loss to new weights"""
        with tf.GradientTape() as tape:
            losses = self.loss(batch)
            gradients = tape.gradient(
                sum(losses.values()), self.trainable_variables
            )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        return losses
