from Environments.GridEnv import GridEnv


class CliffEnv(GridEnv):
    def __init__(self, height=4, width=12):
        super().__init__(height, width)
        self.__start = (self._height - 1, 0)
        self.__goal = (self._height - 1, self._width - 1)
        self.__cliff = [(self._height - 1, j) for j in range(1, self._width - 1)]
        self.__agents_location = None

    def reset(self):
        self.__agents_location = self.__start
        return self.__agents_location

    def step(self, action):
        reward = -1.0
        self.__agents_location = super()._get_new_location(self.__agents_location, action)
        if self.__agents_location in self.__cliff:
            reward = -100.0
            self.__agents_location = self.__start
        done = self.__agents_location == self.__goal
        return self.__agents_location, reward, done

    def _get_location_letter(self, location):
        if location == self.__agents_location:
            return "A"
        if location == self.__goal:
            return "G"
        if location in self.__cliff:
            return "C"
        return " "
