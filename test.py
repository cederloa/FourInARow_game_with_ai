from fourinarowGame import Neljansuora
from aiPlayer import aiPlayer
import itertools


def main():
    gui = Neljansuora()
    p1 = aiPlayer("red")
    p2 = aiPlayer("yellow")

    while gui.voiton_tarkastus() == False:
        p1.set_state(gui.pelilauta())
        p2.set_state(gui.pelilauta())
        if gui.vuorossa() == "p1":
            gui.pelimerkin_pudotus(p1.choose_action())
        else:
            gui.pelimerkin_pudotus(p2.choose_action())

    gui.start()

    p1 = aiPlayer("red")
    p1.set_state(gui.pelilauta())
    print(p1.get_state())
    print(p1.get_available_actions())

if __name__ == '__main__':
    main()
