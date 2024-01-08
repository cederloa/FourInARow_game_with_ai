# An ai player using Q-learning

import numpy as np
import random

class aiPlayer:
    def __init__(self, color):
        self.COLOR_TRANSLATION = {"red": "punainen", "yellow": "keltainen"}
        self.__possible_actions = range(0,7)
        self.__color = self.COLOR_TRANSLATION[color]


    def set_state(self, table):
        state_as_list = []
        for column in table:
            state_column = list(map(lambda x:
                                    0 if x=="tyhja" else 
                                    1 if x==self.__color else
                                    -1, column))
            state_as_list.append(state_column)
        self.__state = np.array(state_as_list)


    def get_state(self):
        return self.__state
    

    def get_available_actions(self):
        return [i for i, col in enumerate(self.__state) if 0 in col]
    

    def choose_action(self):
        # Currently random actions chosen
        # TODO: Implement RL agent (in another file)
        return random.choice(self.get_available_actions())
