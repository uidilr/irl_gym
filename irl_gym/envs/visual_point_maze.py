import cv2
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from irl_gym.envs.dynamic_mjc.mjc_models import point_mass_maze
from irl_gym.envs.env_utils import get_asset_xml

INIT_POS = np.array([0.15,0.15])
TARGET = np.array([0.15, -0.15])
DIST_THRESH = 0.12


class VisualPointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6,
                 sparse_reward=False, no_reward=False, episode_length=100, grayscale=True,
                 width=64, height=64):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length

        self.width = width
        self.height = height
        self.grayscale=grayscale

        self.episode_length = 0

        model = point_mass_maze(direction=self.direction, length=self.length)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        if self.grayscale:
            self.observation_space = Box(0, 1, shape=(width, height))
        else:
            self.observation_space = Box(0, 1, shape=(width, height, 3))

    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_state(self):
        return np.concatenate([
            self.get_body_com("particle"),
            #self.get_body_com("target"),
        ])

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


if __name__ == "__main__":
    from getkey import getkey
    env = VisualPointMazeEnv()

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
