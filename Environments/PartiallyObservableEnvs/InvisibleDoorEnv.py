from Environments.GridEnv import GridEnv


class InvisibleDoorEnv(GridEnv):

    def __init__(self):
        super().__init__(height=4, width=5)
        self.__start = (self._height - 1, 0)
        self.__goal = (0, self._width - 1)
        self.__walls = {(2, 1), (2, 2), (2, 3), (2, 4)}
        self.__button = (3, 4)
        self.__invisible_door = (2, 0)
        self.__time_limit = 5000
        self.__agents_location = None
        self.__is_door_open = None
        self.__current_time = None

    def reset(self):
        self.__current_time = 0
        self.__agents_location = self.__start
        self.__is_door_open = False
        return self.__agents_location

    def step(self, action):
        self.__current_time += 1
        new_location = super()._get_new_location(self.__agents_location, action)
        if self.__is_next_location_valid(new_location):
            self.__agents_location = new_location
        if self.__agents_location == self.__button:
            self.__is_door_open = not self.__is_door_open
        done = self.__agents_location == self.__goal
        reward = 1.0 if done else 0.0
        return self.__agents_location, reward, done or (self.__current_time >= self.__time_limit)

    def __is_next_location_valid(self, next_location):
        return next_location not in self.__walls and (self.__is_door_open or next_location != self.__invisible_door)

    def _get_location_letter(self, location):
        if location == self.__agents_location:
            return "A"
        if location == self.__goal:
            return "G"
        if location == self.__button:
            return "B"
        if location in self.__walls:
            return "X"
        return " "
