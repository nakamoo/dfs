import os
import sys
import time
import cv2
import gym

import numpy as np

class Env(object):
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.real_action = [0, 1, 2, 5]
        self.n_actions = 4
        self._state_buffer = None
        self.render = True

    def reset(self):
        self._state_buffer = None
        return self._preprocess_state(self.env.reset())

    def step(self, action, frameskip=None, eval=False):
        if self.render:
            self.env.render()

        if frameskip is not None:
            self.env.frameskip = frameskip

        state, reward, terminal, info = self.env.step(self.real_action[action])

        # action penalty
        if not eval:
            reward -= 0.3

        return self._preprocess_state(state), reward, terminal, info

    def _preprocess_state(self, state):
        # screen shape is (210, 160, 1)
        gray_observation = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        # resize to height=110, width=84
        # resized_screen = cv2.resize(gray_observation, (84, 110))
        # x_t = resized_screen[18:102, :]
        x_t = cv2.resize(gray_observation, (84, 84))
        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)

        if self._state_buffer is None:
            self._state_buffer = np.stack((x_t, x_t, x_t, x_t))

        x_t = np.reshape(x_t, (1, 84, 84))

        self._state_buffer = np.concatenate((np.delete(self._state_buffer, 0, 0), x_t))
        return self._state_buffer


# class DoomEnv(object):
#
#     def __init__(self, vizdoom_dir=os.path.expanduser('~/ViZDoom'),
#                  window_visible=True, scenario='basic', skipcount=10,
#                  resolution_width=640, sleep=0.0, seed=None):
#
#         self.skipcount = skipcount
#         self.sleep = sleep
#
#         sys.path.append(os.path.join(vizdoom_dir, "examples/python"))
#         from vizdoom import DoomGame
#         from vizdoom import ScreenFormat
#         from vizdoom import ScreenResolution
#
#         game = DoomGame()
#
#         if seed is not None:
#             assert seed >= 0 and seed < 2 ** 16, \
#                 "ViZDoom's random seed must be represented by unsigned int"
#         else:
#             # Use numpy's random state
#             seed = np.random.randint(0, 2 ** 16)
#         game.set_seed(seed)
#
#         # Load a config file
#         game.load_config(os.path.join(
#             vizdoom_dir, "examples", 'config', scenario + '.cfg'))
#
#         # Replace default relative paths with actual paths
#         game.set_vizdoom_path(os.path.join(vizdoom_dir, "bin/vizdoom"))
#         game.set_doom_game_path(
#             os.path.join(vizdoom_dir, 'scenarios/freedoom2.wad'))
#         game.set_doom_scenario_path(
#             os.path.join(vizdoom_dir, 'scenarios', scenario + '.wad'))
#
#         # Set screen settings
#         resolutions = {640: ScreenResolution.RES_640X480,
#                        320: ScreenResolution.RES_320X240,
#                        160: ScreenResolution.RES_160X120}
#         game.set_screen_resolution(resolutions[resolution_width])
#         game.set_screen_format(ScreenFormat.RGB24)
#         game.set_window_visible(window_visible)
#         game.set_sound_enabled(window_visible)
#
#         game.init()
#         self.game = game
#
#         # Use one-hot actions
#         self.n_actions = game.get_available_buttons_size()
#         self.actions = []
#         for i in range(self.n_actions):
#             self.actions.append([i == j for j in range(self.n_actions)])
#
#     def reset(self):
#         self.game.new_episode()
#         return self.game.get_state()
#
#     def step(self, action):
#         r = self.game.make_action(self.actions[action], self.skipcount)
#         r /= 100
#         time.sleep(self.sleep * self.skipcount)
#         return self.game.get_state(), r, self.game.is_episode_finished(), None
