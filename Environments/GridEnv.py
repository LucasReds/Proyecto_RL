import random
from abc import abstractmethod

from Environments.AbstractEnv import AbstractEnv


class GridEnv(AbstractEnv):

    def __init__(self, height, width):
        self._height = height
        self._width = width

    @property
    def action_space(self):
        return ["down", "up", "right", "left"]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def _sample_valid_location(self, invalid_locations):
        location = self.__sample_random_location()
        while location in invalid_locations:
            location = self.__sample_random_location()
        return location

    def __sample_random_location(self):
        i = random.randrange(self._height)
        j = random.randrange(self._width)
        return i, j

    def _get_new_location(self, current_location, action):
        row, col = self.__get_new_raw_location(current_location, action)
        return self.__update_location(row, col)

    @staticmethod
    def __get_new_raw_location(current_location, action):
        row, col = current_location
        if action == "down":
            row += 1
        if action == "up":
            row -= 1
        if action == "right":
            col += 1
        if action == "left":
            col -= 1
        return row, col

    def __update_location(self, new_row, new_col):
        new_row = self.__adjust_limit(new_row, self._height - 1)
        new_col = self.__adjust_limit(new_col, self._width - 1)
        return new_row, new_col

    @staticmethod
    def __adjust_limit(value, max_value):
        return max([min([value, max_value]), 0])

    def show(self):
        print()
        print("X" * (self._width + 2))
        for i in range(self._height):
            print("X", end="")
            for j in range(self._width):
                location = (i, j)
                print(self._get_location_letter(location), end="")
            print("X")
        print("X" * (self._width + 2))

    @abstractmethod
    def _get_location_letter(self, location):
        pass
