from Environments.MultiAgentEnvs.AbstractMultiAgentEnv import AbstractMultiAgentEnv
from Environments.GridEnv import GridEnv


class HunterAndPreyEnv(GridEnv, AbstractMultiAgentEnv):

    def __init__(self):
        super().__init__(height=7, width=7)
        vertical_walls = [(i, j) for i in range(self._height) if i != 3 for j in [0, 6]]
        horizontal_walls = [(i, j) for j in range(self._width) if j != 3 for i in [0, 6]]
        self.__single_agent_actions = super().action_space + ['None']
        self.__walls = set(vertical_walls + horizontal_walls)
        self.__hunter1 = None
        self.__hunter2 = None
        self.__prey = None

    @property
    def action_space(self):
        return [(a1, a2, a3)
                for a1 in self.__single_agent_actions
                for a2 in self.__single_agent_actions
                for a3 in self.__single_agent_actions]

    @property
    def single_agent_action_space(self):
        return self.__single_agent_actions

    @property
    def num_of_agents(self):
        return 3

    def reset(self):
        walls = list(self.__walls)
        self.__hunter1 = self._sample_valid_location(invalid_locations=walls)
        self.__hunter2 = self._sample_valid_location(invalid_locations=[self.__hunter1] + walls)
        self.__prey = self._sample_valid_location(invalid_locations=[self.__hunter1, self.__hunter2] + walls)
        return self.__get_state()

    def __get_state(self):
        return self.__hunter1, self.__hunter2, self.__prey

    def step(self, action):
        hunter1_action, hunter2_action, prey_action = action
        self.__prey = self.get_next_location(self.__prey, prey_action)
        self.__hunter1 = self.get_next_location(self.__hunter1, hunter1_action)
        self.__hunter2 = self.get_next_location(self.__hunter2, hunter2_action)
        done = self.__prey in [self.__hunter1, self.__hunter2]
        hunter_reward = 1.0 if done else 0.0
        prey_reward = -hunter_reward
        return self.__get_state(), (hunter_reward, hunter_reward, prey_reward), done

    def get_next_location(self, current_location, action):
        if action == 'None':
            return current_location
        new_location = self._get_new_location(current_location, action)
        if new_location in self.__walls or new_location in [self.__hunter1, self.__hunter2]:
            return current_location
        return new_location

    def _get_location_letter(self, location):
        if location == self.__hunter1:
            return "A"
        if location == self.__hunter2:
            return "B"
        if location == self.__prey:
            return "o"
        if location in self.__walls:
            return "X"
        return " "
