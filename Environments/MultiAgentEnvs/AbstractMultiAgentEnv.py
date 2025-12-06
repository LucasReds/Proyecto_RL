from abc import ABC, abstractmethod

from Environments.AbstractEnv import AbstractEnv


class AbstractMultiAgentEnv(AbstractEnv, ABC):

    @property
    @abstractmethod
    def single_agent_action_space(self):
        """
        :return: a list with all the actions that are available for a single agent.
        """
        pass

    @property
    @abstractmethod
    def num_of_agents(self):
        """
        :return: returns the number of agents interacting with the environment
        """
        pass
