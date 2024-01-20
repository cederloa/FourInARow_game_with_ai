from fourinarowGui import Gui_4iar
from aiPlayer import aiPlayer
import itertools
import torch
from models.dqn import Dqn


def main():
    RLmodel = torch.load("models/savedModels/firstModel.pt")
    initModel = torch.load("models/savedModels/initModel.pt")

    gui = Gui_4iar()
    p1 = aiPlayer("P1", model=RLmodel)
    p2 = aiPlayer("P2", model="random")

    while gui.getGame().getResults() == False:
        p1.set_state(gui.getGame().getGameBoard())
        p2.set_state(gui.getGame().getGameBoard())
        if gui.getGame().getInturn() == "P1":
            gui.drop(p1.choose_action())
        else:
            gui.drop(p2.choose_action())

    gui.start()

    #print(p1.get_state())
    #print(p1.get_available_actions())

if __name__ == '__main__':
    main()
