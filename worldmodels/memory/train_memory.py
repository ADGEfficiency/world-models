"""
The raw data we use to train the memory are previously generated
means and logvars

We use these rather than the reparameterized latent vector so that
we can sample a different latent vector for a given observation
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from worldmodels.data.tf_records import shuffle_samples, parse_latent_stats
from worldmodels.memory.memory import Memory
from worldmodels.params import memory_params, home

from worldmodels.utils import calc_batch_per_epoch, list_records, make_directories
from worldmodels import setup_logging


def train(model, records, epochs, batch_size, batch_per_epoch, save_every):
    logger = setup_logging(os.path.join(results_dir, 'training.csv'))
    logger.info('epoch, batch, loss')

    dataset = shuffle_samples(
        parse_latent_stats,
        records,
        batch_size=batch_size, shuffle_buffer=500, num_cpu=8
    )

    for epoch in range(epochs):
        batch_loss = np.zeros(batch_per_epoch)
        for batch_num in range(batch_per_epoch):
            batch = next(dataset)
            mu = batch['mu']
            logvars = batch['logvar']
            action = batch['action']

            epsilon = tf.random.normal(shape=mu.shape)
            z = mu + epsilon * tf.exp(logvars * .5)

            x = tf.concat(
                (z[:, :-1, :], action[:, :-1, :]),
                axis=2
            )

            y = z[:, 1:, :]

            assert x.shape[0] == y.shape[0]
            assert y.shape[1] == 999
            assert x.shape[2] == 35
            assert y.shape[2] == 32
            state = model.lstm.get_zero_hidden_state(x)

            batch_loss[batch_num] = model.train_op(x, y, state)

            msg = '{}, {}, {}'.format(
                epoch,
                batch_num,
                batch_loss[batch_num]
            )
            logger.info(msg)

            if batch_num % save_every == 0:
                model.save(results_dir)

        model.save(results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default=0, nargs='?')
    parser.add_argument('--data', default='local', nargs='?')
    parser.add_argument('--cpu', default=8, nargs='?')
    parser.add_argument('--epochs', default=40, nargs='?') # paper says 40
    args = parser.parse_args()

    make_directories('memory-training/models')
    records = list_records('latent-stats', 'episode', args.data)

    results_dir = os.path.join(home, 'memory-training')
    epochs, batch_size, batch_per_epoch = calc_batch_per_epoch(
        epochs=memory_params['epochs'],
        batch_size=memory_params['batch_size'],
        records=records
    )

    memory_params['batch_per_epoch'] = batch_per_epoch
    memory_params['load_model'] = bool(int(args.load_model))
    model = Memory(**memory_params)

    training_params = {
        'records': records,
        'model': model,
        'epochs': args.epochs,
        'batch_size': batch_size,
        'batch_per_epoch': batch_per_epoch,
        'save_every': 20  # batches
    }

    print('cli')
    print('------')
    print(args)
    print('')

    print('training params')
    print('------')
    print(training_params)
    print('')

    print('memory params')
    print('------')
    print(memory_params)
    print('')

    train(**training_params)
