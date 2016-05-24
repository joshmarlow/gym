"""
Incredibly simple environment.
The state is an integer that starts positive and falls by one every timestep.
The agent can respond by adding one of the following to the state:
    -1, 0, 1
A reward of 1 is given for all positive states, and zero for the zero state.
"""
from gym.core import Env
from gym.spaces.discrete import Discrete
import numpy as np


class KeepItPositive(Env):
    max_height = 10
    # three actions - 0, 1, 2
    action_space = Discrete(3)
    observation_space = Discrete(max_height)

    metadata = {
        'render.modes': ['human'],
    }

    def __construct_observation(self, state):
        """Construct an observation from the state."""
        return np.array(state)

    def __compute_reward(self, position):
        return 0.0 if position < 1 else 1

    def _reset(self):
        self.__position = np.random.randint(3, self.max_height)

        return self.__construct_observation(self.__position)

    def _step(self, action):
        # map [0,2] -> [-1,1]
        delta = action - 1
        self.__position = min(self.max_height, max(0, self.__position + delta - 1))

        reward = self.__compute_reward(self.__position)
        done = self.__position <= 0

        return self.__construct_observation(self.__position), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return

        print("Position: {0}, reward: {1}\n".format(self.__position, self.__compute_reward(self.__position)))
