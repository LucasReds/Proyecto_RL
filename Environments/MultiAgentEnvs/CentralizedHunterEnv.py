from Environments.MultiAgentEnvs.AbstractMultiAgentEnv import AbstractMultiAgentEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv


class CentralizedHunterEnv(AbstractMultiAgentEnv):

    def __init__(self):
        self.env = HunterEnv()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_agent_action_space(self):
        return self.env.single_agent_action_space

    @property
    def num_of_agents(self):
        return self.env.num_of_agents

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done = self.env.step(action)
        reward = reward[0]  # In this setting, the rewards are identical, so we only consider one of them
        return state, reward, done

    def show(self):
        self.env.show()
