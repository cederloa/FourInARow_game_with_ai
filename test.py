from fourinarowGui import Gui_4iar
from aiPlayer import aiPlayer
import itertools


def main():
    gui = Gui_4iar()
    p1 = aiPlayer("P1")
    p2 = aiPlayer("P2")

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
