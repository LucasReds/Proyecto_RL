import random

from Environments.MultiAgentEnvs.AbstractMultiAgentEnv import AbstractMultiAgentEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv


class HunterEnv(AbstractMultiAgentEnv):

    def __init__(self):
        self.env = HunterAndPreyEnv()
        self.__state = None
        self.__single_agent_actions = self.env.single_agent_action_space

    @property
    def action_space(self):
        return [(a1, a2) for a1 in self.__single_agent_actions for a2 in self.__single_agent_actions]

    @property
    def num_of_agents(self):
        return 2

    @property
    def single_agent_action_space(self):
        return self.__single_agent_actions

    def reset(self):
        self.__state = self.env.reset()
        return self.__state

    def step(self, action):
        prey_action = self.__sample_prey_action()
        hunter1_action, hunter2_action = action
        self.__state, reward, done = self.env.step((hunter1_action, hunter2_action, prey_action))
        reward = reward[:2]  # we ignore the prey's reward
        return self.__state, reward, done

    def __sample_prey_action(self):
        hunter1, hunter2, prey = self.__state
        best_actions = []
        best_score = float('-inf')
        for action in self.__single_agent_actions:
            next_location = self.env.get_next_location(prey, action)
            distance_hunter1 = self.__compute_distance(next_location, hunter1)
            distance_hunter2 = self.__compute_distance(next_location, hunter2)
            score = min(distance_hunter1, distance_hunter2)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)

    @staticmethod
    def __compute_distance(location1, location2):
        return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])

    def show(self):
        self.env.show()
