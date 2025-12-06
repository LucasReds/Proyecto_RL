from abc import ABC, abstractmethod


class AbstractEnv(ABC):

    @property
    @abstractmethod
    def action_space(self):
        """
        :return: a list with all the actions available in the environment.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment.
        :return: an initial state of the environment.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Performs an action in the environment.
        :param action: is the action to perform in the environment.
        :return: (state, reward, done), where state is the current state, reward is the immediate reward, and
                 done is True iff the environment reached a terminal state.
        """
        pass

    @abstractmethod
    def show(self):
        """
        Shows the current state of the environment in console
        """
        pass