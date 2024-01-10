# Train the Dqn model to play 4 in a row

from aiPlayer import aiPlayer
from fourinarowGame import FourinarowGame
from models.dqn import Dqn
import torch

def trainFromTheBeginning(episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    policyNet = Dqn().to(device)
    targetNet = Dqn().to(device)

    p1 = aiPlayer("P1", policyNet)
    p2 = aiPlayer("P2")



if __name__ == "__main__":
    trainFromTheBeginning()
