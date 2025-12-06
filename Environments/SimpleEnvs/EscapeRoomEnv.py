import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EscapeRoomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, height=10, width=25, max_steps=None):
        super().__init__()

        self.height = height
        self.width = width
        self.max_dist = height + width

        # Curriculum toggle (IMPORTANT)
        self.randomize_layout = False

        if max_steps is None:
            self.max_steps = height * width * 5
        else:
            self.max_steps = max_steps

        # Smaller, stable nail pattern
        self.nails = [(height//2, j) for j in range(3, width-3)]

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(height, width, 5),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_layout:
            # randomized map
            self.agent_pos = (np.random.randint(self.height),
                              np.random.randint(self.width))

            self.key = (np.random.randint(self.height//2),
                        np.random.randint(self.width))

            self.door = (np.random.randint(self.height//2),
                         np.random.randint(self.width))

            while self.key == self.door or self.key == self.agent_pos:
                self.key = (np.random.randint(self.height//2),
                            np.random.randint(self.width))

            while self.door == self.agent_pos:
                self.door = (np.random.randint(self.height//2),
                             np.random.randint(self.width))

        else:
            # FIXED LAYOUT (first training phase)
            self.agent_pos = (self.height - 1, 0)
            self.key = (0, 0)
            self.door = (0, self.width - 1)

        self.has_key = False
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        x, y = self.agent_pos

        prev_key_dist = self._dist(self.agent_pos, self.key)
        prev_door_dist = self._dist(self.agent_pos, self.door)

        # Move agent
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.height - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.width - 1: y += 1

        self.agent_pos = (x, y)

        picked_key = (self.agent_pos == self.key and not self.has_key)
        if picked_key:
            self.has_key = True

        at_nail = self.agent_pos in self.nails
        at_door = (self.agent_pos == self.door)

        terminated = (at_door and self.has_key)
        truncated = (self.current_step >= self.max_steps)

        reward = 0.0
        reward -= 0.001  # small step cost

        # Small exploration bonus (important!)
        if at_door and not self.has_key:
            reward += 0.5

        if picked_key:
            reward += 5.0

        if at_nail:
            reward -= 0.01

        if terminated:
            reward += 50.0

        if truncated:
            reward -= 500.0

        # Normalized shaping
        if not self.has_key:
            shaping = (prev_key_dist - self._dist(self.agent_pos, self.key)) / self.max_dist
        else:
            shaping = (prev_door_dist - self._dist(self.agent_pos, self.door)) / self.max_dist

        reward += shaping * 2.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.zeros((self.height, self.width, 5), dtype=np.float32)

        ax, ay = self.agent_pos
        obs[ax, ay, 0] = 1

        if not self.has_key:
            kx, ky = self.key
            obs[kx, ky, 1] = 1

        dx, dy = self.door
        obs[dx, dy, 2] = 1

        for nx, ny in self.nails:
            obs[nx, ny, 3] = 1

        obs[:, :, 4] = float(self.has_key)

        return obs
