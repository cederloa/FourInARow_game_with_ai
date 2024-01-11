# Train the Dqn model to play 4 in a row

from aiPlayer import aiPlayer
from fourinarowGame import FourinarowGame
from models.dqn import Dqn
import torch
import random


def makeAMove(player, game, E):
    # Get the current player's state of the game
    player.set_state(game.getGameBoard())
    state = player.get_state()

    # Choose action and play it on the gameboard
    isRandom = random.uniform(0, 1) < E
    if not isRandom:
        print("AI CHOSE THIS ACTION!")
    action = player.choose_action(isRandom)
    game.playTurn(action)

    # Get the current player's new state of the game
    player.set_state(game.getGameBoard())
    new_state = player.get_state()

    return game, state, action, new_state


def trainFromTheBeginning(episodes=2):
    E_start = 0.9  # Initial probability to choose random action
    E_end = 0.1  # Final probability to choose random action
    G = 0.9  # Discount factor
    T = 0.1  # Update rate
    LR = 1e-3  # Learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policyNet = Dqn().to(device)  # Updated after all episodes, and saved (?) Moves are made based on this.
    targetNet = Dqn().to(device)  # Updated after every episode, based on policyNet (?)

    p1 = aiPlayer("P1", policyNet)
    p2 = aiPlayer("P2", policyNet)

    # Load the policyNet state to targetNet
    targetNet.load_state_dict(policyNet.state_dict())

    optimizer = torch.optim.AdamW(policyNet.parameters(), lr=LR)

    for ep in range(episodes):
        # Every new episode is a new game
        game = FourinarowGame(p1.get_id(), p2.get_id())

        # Play the game until it ends
        while game.getResults() == False:
            if game.getInturn() == p1.get_id():
                game, state, action, new_state = makeAMove(p1, game, E_start)
            else:
                game, state, action, new_state = makeAMove(p2, game, E_start)

            print(f"{state}\n{action}\n{new_state}\n")

            # TODO: Save state, action and new_state for training

            

if __name__ == "__main__":
    trainFromTheBeginning()
