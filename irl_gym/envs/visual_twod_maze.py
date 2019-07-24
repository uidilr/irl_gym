import cv2
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from irl_gym.envs.env_utils import get_asset_xml

INIT_POS = np.array([0.15,0.15])
TARGET = np.array([0.15, -0.15])
DIST_THRESH = 0.12


class VisualTwoDMaze(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, verbose=False, width=64, height=64, grayscale=True):
        self.verbose = verbose
        self.max_episode_length = 200
        self.episode_length = 0
        self.width = width
        self.height = height
        self.grayscale = grayscale
        utils.EzPickle.__init__(self)
        super().__init__(get_asset_xml('twod_maze.xml'), frame_skip=2)

        self.observation_space = Box(0, 1, shape=(width, height, 3))

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        state = self._get_state()
        pos = state[0:2]
        dist = np.sum(np.abs(pos-TARGET)) #np.linalg.norm(pos - TARGET)
        reward = - (dist)

        reward_ctrl = - np.square(a).sum()
        reward += 1e-3 * reward_ctrl

        if self.verbose:
            print(pos, reward)
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length

        ob = self._get_obs()
        return ob, reward, done, {'distance': dist}

    def reset_model(self):
        self.episode_length = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_state(self):
        return np.concatenate([self.data.qpos]).ravel() - INIT_POS

    def _get_obs(self):
        viewer = self._get_viewer(mode='human')
        viewer.render()
        window_context = viewer.opengl_context
        width, height = window_context._width, window_context._height
        image = self.render(mode='rgb_array', width=width, height=height)

        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)/255.0
        return image

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0


if __name__ == "__main__":
    from getkey import getkey
    env = VisualTwoDMaze()

    while True:
        key = getkey()
        a = np.array([0.0,0.0])
        if key == 'w':
            a += np.array([0.0, 1.0])
        elif key == 'a':
            a += np.array([-1.0, 0.0])
        elif key  == 's':
            a += np.array([0.0, -1.0])
        elif key  == 'd':
            a += np.array([1.0, 0.0])
        elif key  == 'q':
            break
        a *= 0.2
        o, r,_,_ = env.step(a)
        print(r)