"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from data_collector import DataCollector
from math import sqrt
from gym.envs.classic_control import rendering

#create function to calculate Manhattan distance

class AimBotWithResetEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, resolution, accuracy):
        height, width = resolution
        self.min_position_y = 0
        self.max_position_y = height
        self.min_position_x = 0
        self.max_position_x = width
        self.tolerance_x = accuracy
        self.tolerance_y = accuracy

        self.delta_y = 0.001
        self.delta_x = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position_x, -self.min_position_y])
        self.high = np.array([self.max_position_x, self.max_position_y])

        self.viewer = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def manhattan(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # self.step is ((x,y),(x,y))
        reticle, target = self.state
        if action == 0:
            reticle[0] += self.delta_x
            target[0] += self.delta_x
        elif action == 1:
            reticle[1] += self.delta_y
            target[1] += self.delta_y
        elif action == 2:
            reticle[0] -= self.delta_x
            target[0] -= self.delta_x
        elif action == 3:
            reticle[1] -= self.delta_y
            target[1] -= self.delta_y

        diff_x = reticle[0] - target[0]
        diff_y = reticle[1] - target[1]
        if target[0] < self.min_position_x or target[0] > self.max_position_x or target[0] < self.min_position_x or target[0] > self.max_position_x:
            done = True
            reward = float(done) * -100
        elif diff_x <= self.tolerance_x and diff_y <= self.tolerance_y:
            done = True
            reward = float(done) * 100
        else:
            done = False
            reward = self.manhattan(reticle, target)
        self.state = [reticle, target]
        return np.array(self.state), reward, done, {}

    def reset(self):
        target = (self.np_random.uniform(low=0, high=self.high[0]), self.np_random.uniform(low=0, high=self.high[1]))
        reticle = (self.np_random.uniform(low=0, high=self.high[0]), self.np_random.uniform(low=0, high=self.high[1]))
        return self.reset_specific(reticle, target)

    def reset_specific(self, reticle_position, target_position):
        assert self.min_position_x <= target_position[0] <= self.max_position_x
        assert self.min_position_x <= reticle_position[0] <= self.max_position_x
        assert self.min_position_y <= target_position[1] <= self.max_position_y
        assert self.min_position_y <= target_position[1] <= self.max_position_y
        self.state = (reticle_position, target_position)
        return np.array(self.state).flatten()

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = self.max_position_x
        screen_height = self.max_position_y
        reticle_width = 40
        reticle_height = 20
        target_width = 40
        target_height = 20
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)  # Create a Viewer object

            # Clear viewer
        self.viewer.clear()

        # Create a white background
        background = rendering.make_polygon([(0, 0), (0, 500), (500, 500), (500, 0)], filled=True)
        background.set_color(1, 1, 1)  # White color
        self.viewer.add_geom(background)

        # Get current state information
        # For example, let's say state is represented by (x, y) coordinates of two symbols
        symbol1_x, symbol1_y = self.state[0]
        symbol2_x, symbol2_y = self.state[1]

        # Draw symbols on the viewer
        symbol_size = 10  # Size of the symbols
        symbol1 = rendering.make_circle(symbol_size)
        symbol1.set_color(0, 0, 0)  # Black color
        symbol1.add_attr(rendering.Transform(translation=(symbol1_x, symbol1_y)))
        self.viewer.add_geom(symbol1)

        symbol2 = rendering.make_circle(symbol_size)
        symbol2.set_color(0, 0, 0)  # Black color
        symbol2.add_attr(rendering.Transform(translation=(symbol2_x, symbol2_y)))
        self.viewer.add_geom(symbol2)

        return self.viewer.render(return_rgb_array=True)

    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = MountainCarWithResetEnv()
    print(env)
    #Collecting 100K data samples
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(10)
    print(actions)
    state = env.reset_specific(0.5, 0.0)
    next_state, reward, done, _ = env.step(1)
    print(reward)
    # env.render()
    is_done = False
    # while not is_done:
    #     _, r, is_done, _ = env.step(2)  # go left
    #     env.render()
    #     print(r)
    # env.close()
