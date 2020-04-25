import argparse
import os

import tensorflow as tf

from worldmodels import setup_logging
from worldmodels.data.tf_records import parse_episode, shuffle_samples
from worldmodels.params import vae_params, results_dir
from worldmodels.vision.vae import VAE
from worldmodels.vision.images import compare_images, generate_images, generate_gif
from worldmodels.utils import calc_batch_per_epoch, list_records, make_directories


def train(model, records, epochs, batch_size, log_every, save_every):
    logger = setup_logging(os.path.join(results_dir, 'training.csv'))
    logger.info('epoch, batch, reconstruction-loss, kl-loss')

    dataset = shuffle_samples(parse_episode, records, batch_size)
    sample_observations = next(dataset)[0]

    sample_observations = sample_observations.numpy()[:4]
    sample_latent = tf.random.normal(shape=(4, model.latent_dim))

    epochs, batch_size, batch_per_epoch = calc_batch_per_epoch(
        epochs=epochs,
        batch_size=batch_size,
        records=records,
        samples_per_record=1000
    )

    image_dir = os.path.join(results_dir, 'images')
    for epoch in range(epochs):
        generate_images(model, epoch, 0, sample_latent, image_dir)

        for batch_num in range(batch_per_epoch):

            batch, _ = next(dataset)
            losses = model.backward(batch)

            msg = '{}, {}, {}, {}'.format(
                epoch,
                batch_num,
                losses['reconstruction-loss'].numpy(),
                losses['kl-loss'].numpy(),
            )
            logger.info(msg)

            if batch_num % log_every == 0:
                print(msg)

            if batch_num % save_every == 0:
                model.save(results_dir)
                generate_images(model, epoch, batch_num, sample_latent, image_dir)
                compare_images(model, sample_observations, results_dir)
                generate_gif(image_dir, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', default=1, nargs='?')
    parser.add_argument('--log_every', default=100, nargs='?')
    parser.add_argument('--save_every', default=1000, nargs='?')
    parser.add_argument('--epochs', default=10, nargs='?')
    parser.add_argument('--data', default='local', nargs='?')
    parser.add_argument('--dataset', default='random-rollouts', nargs='?')
    args = parser.parse_args()

    make_directories('vae-training/images')
    results_dir = os.path.join(results_dir, 'vae-training')

    if args.dataset == 'random':
        records = list_records(
            path='random-rollouts',
            contains='episode',
            data=args.data
        )

    else:
        records = list_records(
            path='controller-rollouts',
            contains='episode',
            data=args.data
        )

    vae_params['load_model'] = bool(int(args.load_model))
    model = VAE(**vae_params)

    training_params = {
        'model': model,
        'epochs': int(args.epochs),
        'batch_size': 256,
        'log_every': int(args.log_every),  # batches
        'save_every': int(args.save_every),  # batches
        'records': records
    }

    print('cli')
    print('------')
    print(args)
    print('')

    print('training params')
    print('------')
    print(training_params)
    print('')

    print('vision params')
    print('------')
    print(vae_params)
    print('')

    train(**training_params)
