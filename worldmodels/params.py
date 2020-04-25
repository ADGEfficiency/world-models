from os import environ, path


home = path.join(environ['HOME'], 'world-models-experiments')

env_params = {
    'num_actions': 3
}

memory_params = {
    'input_dim': 35,  # latent_size + num_actions
    'output_dim': 32,  # latent size
    'num_timesteps': 999,
    'batch_size': 100,
    'lstm_nodes': 256,
    'num_mix': 5,
    'grad_clip': 1.0,
    'initial_learning_rate': 0.001,
    'end_learning_rate': 0.00001,
    'load_model': True,
    'results_dir': path.join(home, 'memory-training')
}

vae_params = {
    'latent_dim': 32,
    'learning_rate': 0.0001,
    'load_model': True,
    'results_dir': path.join(home, 'vae-training')
}
