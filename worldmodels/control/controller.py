import numpy as np


def get_action(z, state, params):
    """ takes an action based on z, h and controller params """
    w, b = shape_controller_params(params)
    net_input = np.concatenate([z, state], axis=None)
    action = np.tanh(net_input.dot(w) + b)
    action[1] = (action[1] + 1.0) / 2.0
    action[2] = np.clip(action[2], 0.0, 1.0)
    return action.astype(np.float32)


def shape_controller_params(params, output_size=3):
    """ split into weights & biases """
    w = params[:-output_size].reshape(-1, output_size)
    b = params[-output_size:]
    return w, b
