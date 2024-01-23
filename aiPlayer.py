# An ai player using Q-learning

import numpy as np
import random
from models.dqn import Dqn
import torch

class aiPlayer:
    def __init__(self, id, model="random"):
        self.__possible_actions = range(0,7)
        self.__id = id
        self.__model_name = None
        self.__known_models = {"dqn": None, "random": self.randomModel}

        # Check if given model is valid
        if isinstance(model, Dqn):
            self.__model = model
        elif model not in self.__known_models:
            raise ValueError(f"Model \"{model}\" not found.")
        elif self.__known_models[model] == None:
            raise NotImplementedError(f"Model \"{model}\" not implemented.")
        else:
            self.__model_name = model
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
    

    def get_id(self):
        return self.__id
    

    def get_model_name(self):
        return self.__model_name
    

    def randomModel(self):
        return random.choice(self.get_available_actions())
    

    def choose_action(self, rnd=False):
        # Currently random actions chosen
        # TODO: Implement RL agent (in another file)

        # Even if the player's model is not random, the player can choose a
        # random (off-policy) action.
        if rnd:
            return self.randomModel()

        # Random model returns an action, but dqn returns an one-hot encoding of
        # all actions. Perform differently based on which model is used.
        if self.__model == self.randomModel:
            return self.__model()
        else:
            # Format the state tensor and feed it to the model
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
            st_tensor = torch.from_numpy(self.__state).float().to(device)
            st_tensor = st_tensor[None, :]  # Adding channel dim
            oneHot_actions = self.__model(st_tensor).cpu().detach().numpy()

            # TODO: THIS NOW CHOOSES THE WRONG ACTION BECAUSE E.G. IF AVAILABLE
            # ACTIONS ARE [0, 1, 2, 4, 5, 6], INDEX 5 MEANS ACTION 6

            ar_dict = {k: v for k, v in enumerate(oneHot_actions)
                       if k in self.get_available_actions()}
            max_a = max(ar_dict, key=ar_dict.get)

            return max_a


class player:
    def __init__(self, id):
        self.__possible_actions = range(0,7)
        self.__id = id


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
    

    def get_id(self):
        return self.__id
    

    def randomModel(self):
        return random.choice(self.get_available_actions())
    

    def choose_action(self, rnd=False):
        return self.randomModel()