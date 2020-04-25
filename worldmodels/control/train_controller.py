from collections import defaultdict
import logging
from multiprocessing import Pool
import os
import pickle

import numpy as np

from worldmodels.control.controller import get_action
from worldmodels.data.car_racing import CarRacingWrapper
from worldmodels.params import vae_params, memory_params, env_params, home


def make_logger(name):
    """ sets up experiment logging to a file """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fldr = os.path.join(home, 'control')
    os.makedirs(fldr, exist_ok=True)
    fh = logging.FileHandler(os.path.join(fldr, '{}.log'.format(name)))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def episode(params, seed, collect_data=False, episode_length=1000, render=False):
    """ runs a single episode """
    #  needs to be imported here for multiprocessing
    import tensorflow as tf
    from worldmodels.vision.vae import VAE
    from worldmodels.memory.memory import Memory

    vision = VAE(**vae_params)

    memory_params['num_timesteps'] = 1
    memory_params['batch_size'] = 1
    memory = Memory(**memory_params)

    state = memory.lstm.get_zero_hidden_state(
        np.zeros(35).reshape(1, 1, 35)
    )

    env = CarRacingWrapper(seed=seed)
    total_reward = 0
    data = defaultdict(list)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(episode_length):
        if render:
            env.render("human")
        obs = obs.reshape(1, 64, 64, 3).astype(np.float32)
        mu, logvar = vision.encode(obs)
        z = vision.reparameterize(mu, logvar)

        action = get_action(z, state[0], params)
        obs, reward, done, _ = env.step(action)

        x = tf.concat([
            tf.reshape(z, (1, 1, 32)),
            tf.reshape(action, (1, 1, 3))
        ], axis=2)

        y, h_state, c_state = memory(x, state, temperature=1.0)
        state = [h_state, c_state]
        total_reward += reward

        if done:
            print('done at {} - reward {}'.format(step, reward))
            break

        if collect_data:
            reconstruct = vision.decode(z)
            vae_loss = vision.loss(reconstruct)
            data['observation'].append(obs)
            data['latent'].append(np.squeeze(z))
            data['reconstruct'].append(np.squeeze(reconstruct))
            data['reconstruction-loss'].append(vae_loss['reconstruction-loss']),
            data['unclipped-kl-loss'].append(vae_loss['unclipped-kl-loss'])
            data['action'].append(action)
            data['mu'].append(mu)
            data['logvar'].append(logvar)
            data['pred-latent'].append(y)
            data['pred-reconstruct'].append(np.squeeze(vision.decode(y.reshape(1, 32))))
            data['total-reward'].append(total_reward)

    env.close()
    logger.debug(total_reward)
    return total_reward, params, data


class CMAES:
    def __init__(self, x0, s0=0.5, opts={}):
        """ wrapper around cma.CMAEvolutionStrategy """
        self.num_parameters = len(x0)
        print('{} params in controller'.format(self.num_parameters))
        self.solver = CMAEvolutionStrategy(x0, s0, opts)

    def __repr__(self):
        return '<pycma wrapper>'

    def ask(self):
        """ sample parameters """
        samples = self.solver.ask()
        return np.array(samples).reshape(-1, self.num_parameters)

    def tell(self, samples, fitness):
        """ update parameters with total episode reward """
        return self.solver.tell(samples, -1 * fitness)

    @property
    def mean(self):
        return self.solver.mean


logger = make_logger('all-rewards')
global_logger = make_logger('rewards')


if __name__ == '__main__':
    generations = 500
    popsize = 64
    epochs = 16

    results_dir = os.path.join(home, 'control', 'generations')
    os.makedirs(results_dir, exist_ok=True)

    #  need to open the Pool before importing from cma
    with Pool(popsize, maxtasksperchild=32) as p:
        from cma import CMAEvolutionStrategy

        input_size = vae_params['latent_dim'] + memory_params['lstm_nodes']
        output_size = env_params['num_actions']

        weights = np.random.randn(input_size, output_size)
        biases = np.random.randn(output_size)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])

        previous_gens = os.listdir(results_dir)
        sort_idx = [int(s.split('_')[1]) for s in previous_gens]
        previous_gens = [p for (i, p) in sorted(zip(sort_idx, previous_gens))]

        if len(previous_gens) > 0:
            previous_gen = previous_gens[-1]
            start_generation = int(previous_gen.split('_')[-1]) + 1

            with open(os.path.join(results_dir, previous_gen, 'es.pkl'), 'rb') as save:
                es = pickle.load(save)
                print('loaded from previous generation {}'.format(previous_gen))

        else:
            es = CMAES(x0, opts={'popsize': popsize})
            start_generation = 0

        print('starting from generation {}'.format(start_generation))
        for generation in range(start_generation, generations):
            population = es.ask()

            epoch_results = np.zeros((popsize, epochs))
            for epoch in range(epochs):
                seeds = np.random.randint(
                    low=0, high=10000,
                    size=population.shape[0]
                )

                results = p.starmap(episode, zip(population, seeds))
                rew, para, data = zip(*results)
                epoch_results[:, epoch] = rew

            epoch_results = np.mean(epoch_results, axis=1)
            assert epoch_results.shape[0] == popsize
            global_logger.debug(np.mean(epoch_results))

            es.tell(para, epoch_results)

            best_params_idx = np.argmax(epoch_results)
            best_params = population[best_params_idx]
            gen_dir = os.path.join(results_dir, 'generation_{}'.format(generation))
            os.makedirs(gen_dir, exist_ok=True)

            np.save(os.path.join(gen_dir, 'population-params.npy'), population)
            np.save(os.path.join(gen_dir, 'best-params.npy'), best_params)
            np.save(os.path.join(gen_dir, 'epoch-results.npy'), epoch_results)

            with open(os.path.join(gen_dir, 'es.pkl'), 'wb') as save:
                pickle.dump(es, save)
