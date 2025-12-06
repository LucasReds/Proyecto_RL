import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EscapeRoomEnvVector(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, height=10, width=10, max_steps=None, randomize_layout=False):
        super().__init__()

        self.height = height
        self.width = width
        self.randomize_layout = randomize_layout

        if max_steps is None:
            self.max_steps = height * width * 5
        else:
            self.max_steps = max_steps

        # Observation: 9 floats
        # agent_x, agent_y,
        # key_x, key_y,
        # door_x, door_y,
        # has_key,
        # dist_to_key, dist_to_door
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)
        self._rng = np.random.default_rng()

    def _normalize(self, pos):
        return np.array([pos[0] / self.height, pos[1] / self.width], dtype=np.float32)

    def _manhattan(self, a, b):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / (self.height + self.width)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_layout:
            self.agent_pos = (
                self._rng.integers(self.height),
                self._rng.integers(self.width),
            )
            self.key = (
                self._rng.integers(self.height),
                self._rng.integers(self.width),
            )
            self.door = (
                self._rng.integers(self.height),
                self._rng.integers(self.width),
            )
        else:
            # fixed layout
            self.agent_pos = (self.height - 1, 0)
            self.key = (0, 0)
            self.door = (0, self.width - 1)

        self.has_key = False
        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        ax, ay = self._normalize(self.agent_pos)
        kx, ky = self._normalize(self.key)
        dx, dy = self._normalize(self.door)

        d_key = self._manhattan(self.agent_pos, self.key)
        d_door = self._manhattan(self.agent_pos, self.door)

        return np.array([
            ax, ay,
            kx, ky,
            dx, dy,
            float(self.has_key),
            d_key,
            d_door
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        x, y = self.agent_pos

        # Move
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.height - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.width - 1: y += 1

        self.agent_pos = (x, y)

        reward = -0.001
        terminated = False
        truncated = self.current_step >= self.max_steps

        # pick up key
        if self.agent_pos == self.key and not self.has_key:
            self.has_key = True
            reward += 3.0

        # door interaction
        if self.agent_pos == self.door:
            if self.has_key:
                reward += 50.0
                terminated = True
            else:
                reward += 0.3

        return self._get_obs(), reward, terminated, truncated, {}
