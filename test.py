from fourinarowGui import Gui_4iar
from aiPlayer import aiPlayer
import itertools
import torch
from models.dqn import Dqn

# NOTE:
# The model does not know how to avoid losing. Find a way to give -1 rewards


def main():
    #RLmodel1 = torch.load("models/savedModels/12channels/9step_100ep_model.pt")
    RLmodel2 = torch.load("models/savedModels/simpleNet/90step_100ep_model.pt")

    gui = Gui_4iar()
    p1 = aiPlayer("P1", model=RLmodel2)
    p2 = aiPlayer("P2", model=RLmodel2)

    while gui.getGame().getResults() == False:
        p1.set_state(gui.getGame().getGameBoard())
        p2.set_state(gui.getGame().getGameBoard())
        if gui.getGame().getInturn().get_id() == "P1":
            gui.drop(p1.choose_action())
        else:
            gui.drop(p2.choose_action())

    gui.start()

    #print(p1.get_state())
    #print(p1.get_available_actions())

if __name__ == '__main__':
    main()
