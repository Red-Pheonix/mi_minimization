# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect


class ColoredFloor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self):
        super().__init__("floor", "blue")
        self.color = np.array([220, 220, 30])

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a heatmapped color
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), self.color)

    def encode(self) -> tuple[int, int, int]:
        return (100, 255, 0)


class ImgReshapeWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = (3,7,7)
        self.observation_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.spaces["image"].low), shape),
            high=np.reshape(np.ravel(env.observation_space.spaces["image"].high), shape),
            shape=shape,
            dtype=env.observation_space.spaces["image"].dtype,
        )

    def observation(self, obs):
        return obs["image"].reshape(3,7,7)

class DataInjectionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DataInjectionWrapper, self).__init__(env)

    def step(self, action, agent_data=None):
        observation, reward, done, trunc, info = self.env.step(action)
        agent_state = tuple(self.env.unwrapped.agent_pos) + (self.env.unwrapped.agent_dir,)
        info["agent_state"] = np.array(agent_state)
        self.env.unwrapped.grid.set(agent_state[0],agent_state[1], ColoredFloor())
        return observation, reward, done, trunc, info

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ImgReshapeWrapper(env)
        env = DataInjectionWrapper(env)
        return env

    return thunk