import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class EscapeRoomEnvExtended(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, height=10, width=10, max_steps=None,
                 randomize_layout=False, n_walls=0, shaped_rewards=True,
                 ensure_solvable=False):
        super().__init__()

        self.height = height
        self.width = width
        self.randomize_layout = randomize_layout
        self.n_walls = n_walls
        self.shaped_rewards = shaped_rewards
        self.ensure_solvable = ensure_solvable

        if max_steps is None:
            self.max_steps = height * width * 3
        else:
            self.max_steps = max_steps

        # 9-dimensional observation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng()

    # UTILS

    def _normalize(self, pos):
        """Return normalized (x,y)."""
        return np.array([
            pos[0] / (self.height - 1),
            pos[1] / (self.width - 1)
        ], dtype=np.float32)

    def _manhattan(self, a, b):
        """Return normalized Manhattan distance."""
        max_dist = (self.height - 1) + (self.width - 1)
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / max_dist

    def _generate_walls(self):
        """Generate walls randomly. If ensure_solvable=True, regenerate until path exists."""
        while True:
            self.walls = set()
            for _ in range(self.n_walls):
                w = (self._rng.integers(self.height),
                     self._rng.integers(self.width))
                if w not in [self.agent_pos, self.key, self.door]:
                    self.walls.add(w)

            if not self.ensure_solvable:
                break
            if self._path_exists(self.agent_pos, self.key) and \
               self._path_exists(self.key, self.door):
                break

    def _path_exists(self, start, goal):
        """ BFS to ensure solvability."""
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True
            for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    if (nx, ny) not in self.walls and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return False

    # RESET
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_layout:
            self.agent_pos = (self._rng.integers(self.height),
                              self._rng.integers(self.width))
            self.key = (self._rng.integers(self.height),
                        self._rng.integers(self.width))
            self.door = (self._rng.integers(self.height),
                         self._rng.integers(self.width))
        else:
            self.agent_pos = (self.height - 1, 0)
            self.key = (0, 0)
            self.door = (0, self.width - 1)

        self.has_key = False
        self.current_step = 0
        self.prev_d_key = self._manhattan(self.agent_pos, self.key)
        self.prev_d_door = self._manhattan(self.agent_pos, self.door)

        # Walls
        if self.n_walls > 0:
            self._generate_walls()
        else:
            self.walls = set()

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

    # STEP
    def step(self, action):
        self.current_step += 1
        reward = -0.005  # Slight step penalty to prevent wandering

        x, y = self.agent_pos
        nx, ny = x, y

        if action == 0 and x > 0: nx -= 1
        elif action == 1 and x < self.height - 1: nx += 1
        elif action == 2 and y > 0: ny -= 1
        elif action == 3 and y < self.width - 1: ny += 1

        # Wall collision, stay still
        if (nx, ny) not in self.walls:
            self.agent_pos = (nx, ny)

        # Reward shaping
        new_d_key = self._manhattan(self.agent_pos, self.key)
        new_d_door = self._manhattan(self.agent_pos, self.door)

        if self.shaped_rewards:
            # Reward for approaching the key or door
            reward += 0.2 * (self.prev_d_key - new_d_key)
            if self.has_key:
                reward += 0.2 * (self.prev_d_door - new_d_door)

        self.prev_d_key = new_d_key
        self.prev_d_door = new_d_door

        # Key pickup
        if self.agent_pos == self.key and not self.has_key:
            self.has_key = True
            reward += 3.0

        # Door interaction
        terminated = False
        if self.agent_pos == self.door:
            if self.has_key:
                reward += 50.0
                terminated = True
            else:
                reward -= 0.1  # DOOR WITHOUT KEY = BAD

        # Truncation
        truncated = self.current_step >= self.max_steps
        if truncated:
            reward -= 1.0 

        return self._get_obs(), reward, terminated, truncated, {}
