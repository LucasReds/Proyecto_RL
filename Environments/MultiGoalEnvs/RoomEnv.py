import random

from Environments.GridEnv import GridEnv


class RoomEnv(GridEnv):

    def __init__(self):
        super().__init__(height=11, width=17)
        self.__start = (2, 0)
        vertical_walls = [(i, j) for i in range(self._height) for j in [5, 10]]
        horizontal_walls = [(5, j) for j in range(5)] + [(6, j) for j in range(6, 10)] + [(5, j) for j in range(11, 17)]
        doors = [(1, 5), (9, 5), (3, 10), (8, 10), (5, 2), (6, 8), (5, 14)]
        self.__walls = set(vertical_walls + horizontal_walls)
        for door in doors:
            self.__walls.remove(door)
        self.__goals = doors + [(0, 0), (10, 0), (0, 16), (10, 16)]
        self.__nonempty_locations = self.__goals + list(self.__walls)
        self.__agents_location = None
        self.__current_goal = None

    @property
    def goals(self):
        return self.__goals

    def reset(self):
        self.__agents_location = self._sample_valid_location(invalid_locations=self.__nonempty_locations)
        self.__current_goal = random.choice(self.__goals)
        return self.__get_state()

    def __get_state(self):
        return self.__agents_location, self.__current_goal

    def step(self, action):
        new_location = super()._get_new_location(self.__agents_location, action)
        if new_location not in self.__walls:
            self.__agents_location = new_location
        done = self.__agents_location == self.__current_goal
        reward = 1.0 if done else 0.0
        return self.__get_state(), reward, done

    def _get_location_letter(self, location):
        if location == self.__agents_location:
            return "A"
        if location == self.__current_goal:
            return "G"
        if location in self.__walls:
            return "X"
        return " "
