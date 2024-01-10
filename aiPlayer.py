# An ai player using Q-learning

import numpy as np
import random
from models.dqn import Dqn

class aiPlayer:
    def __init__(self, id, model="random"):
        self.__possible_actions = range(0,7)
        self.__id = id
        self.__known_models = {"dqn": None, "random": self.randomModel}

        # Check if given model is valid
        if isinstance(model, Dqn):
            self.__model = model
        if model not in self.__known_models:
            raise ValueError(f"Model \"{model}\" not found.")
        elif self.__known_models[model] == None:
            raise NotImplementedError(f"Model \"{model}\" not implemented.")
        else:
            self.__model = self.__known_models[model]


    def set_state(self, table):
        state_as_list = []
        for column in table:
            state_column = list(map(lambda x:
                                    0 if x==None else 
                                    1 if x==self.__id else
                                    -1, column))
            state_as_list.append(state_column)
        self.__state = np.array(state_as_list)


    def get_state(self):
        return self.__state
    

    def get_available_actions(self):
        # Maybe move this to fourinarowGame
        return [i for i, col in enumerate(self.__state) if 0 in col]
    

    def randomModel(self):
        return random.choice(self.get_available_actions())
    

    def choose_action(self):
        # Currently random actions chosen
        # TODO: Implement RL agent (in another file)
        return self.__model()
