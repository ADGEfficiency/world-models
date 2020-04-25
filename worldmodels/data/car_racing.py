from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
import numpy as np
from PIL import Image


def process_frame(
    frame,
    screen_size=(64, 64),
    vertical_cut=84,
    max_val=255,
    save_img=False
):
    """ crops, scales & convert to float """
    frame = frame[:vertical_cut, :, :]
    frame = Image.fromarray(frame, mode='RGB')
    obs = frame.resize(screen_size, Image.BILINEAR)
    return np.array(obs) / max_val


class CarRacingWrapper(CarRacing):
    screen_size = (64, 64)

    def __init__(self, seed=None):
        super().__init__()
        if seed:
            self.seed(int(seed))

        #  new observation space to deal with resize
        self.observation_space = Box(
                low=0,
                high=255,
                shape=self.screen_size + (3,)
        )

    def step(self, action, save_img=False):
        """ one step through the environment """
        frame, reward, done, info = super().step(action)

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        obs = process_frame(
            frame,
            self.screen_size,
            vertical_cut=84,
            max_val=255.0,
            save_img=save_img
        )
        return obs, reward, done, info

    def reset(self):
        """ resets and returns initial observation """
        raw = super().reset()

        #  needed to get image rendering
        #  https://github.com/openai/gym/issues/976
        self.viewer.window.dispatch_events()

        return process_frame(
            raw,
            self.screen_size,
            vertical_cut=84,
            max_val=255.0,
            save_img=False
        )
