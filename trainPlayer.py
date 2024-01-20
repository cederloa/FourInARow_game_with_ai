# Train the Dqn model to play 4 in a row

from aiPlayer import aiPlayer
from fourinarowGame import FourinarowGame
from models.dqn import Dqn
import torch
import random
import numpy as np
from matplotlib import pyplot as plt


def makeAMove(player, game, E):
    # Get the current player's state of the game
    player.set_state(game.getGameBoard())
    state = player.get_state()

    # Choose action and play it on the gameboard
    isRandom = random.uniform(0, 1) < E
    action = player.choose_action(isRandom)
    game.playTurn(action)

    # Get the current player's new state of the game
    player.set_state(game.getGameBoard())
    new_state = player.get_state()

    return game, state, action, new_state


def trainLoop(policyNet=None, episodes=2, E=0.5):
    G = 0.9  # Discount factor
    T = 0.1  # Update rate
    LR = 1e-3  # Learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policyNet == None:
        policyNet = Dqn()  # Updated after all episodes, and saved (?) Moves are made based on this.
    policyNet.to(device)
    targetNet = Dqn().to(device)  # Updated after every episode, based on policyNet (?)

    # Load the policyNet state to targetNet
    targetNet.load_state_dict(policyNet.state_dict())

    p1 = aiPlayer("P1", policyNet)
    p2 = aiPlayer("P2", policyNet)

    optimizer = torch.optim.AdamW(policyNet.parameters(), lr=LR)
    #lossFunction = torch.nn.SmoothL1Loss()
    lossFunction = torch.nn.MSELoss()
    train_losses = []

    # Play the game for a certain amount of episodes (1 episode = 1 game) and
    # save the states, actions and rewards for training
    for ep in range(episodes):
        sa_memory = []  # For saving states and actions
        game = FourinarowGame(p1.get_id(), p2.get_id())
        # Play the game until it ends
        while game.getResults() == False:
            if game.getInturn() == p1.get_id():
                game, state, action, new_state = makeAMove(p1, game, E)
            else:
                game, state, action, new_state = makeAMove(p2, game, E)

            if game.getResults() == False or game.getResults() == "Draw":
                reward = 0
            else:
                reward = 1

            sa_memory.append({"state":state,
                              "action":action,
                              "new_state":new_state,
                              "reward":reward})
        
    
        # Optimize model
        epochs = 10
        batch_size = 1
        temp_losses = []
        for epoch in range(epochs):
            batch = random.sample(sa_memory, batch_size)
            #batch = [sa_memory[-1]]
            #print(batch[0]["state"])

            # Prepare samples for DQN
            st_tensor = torch.from_numpy(batch[0]["state"]).float().to(device)
            st_tensor = st_tensor[None, :]
            newSt_tensor = torch.from_numpy(batch[0]["new_state"]).float().to(device)
            newSt_tensor = newSt_tensor[None, :]
            
            # TODO: Change the loss so that it is localized for the taken action


            # Q-value for the current state
            old_Q = policyNet(st_tensor)[batch[0]["action"]]
            # New Q calculated with r and max Q at next state
            # (Target net: "ground truth")
            new_Q = batch[0]["reward"] + G*torch.max(targetNet(newSt_tensor))

            #old_Q_array = np.zeros(7)
            #old_Q_array[batch[0]["action"]] = old_Q
            #new_Q_array = np.zeros(7)
            #new_Q_array[batch[0]["action"]] = new_Q


            #print(f"old: {old_Q}")
            #print(f"new: {new_Q}")
            loss = lossFunction(old_Q, new_Q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_losses.append(loss.cpu().detach().numpy())
        train_losses.append(np.mean(temp_losses))

    # Visualize losses
    print(len(train_losses))
    plt.plot(train_losses)
    plt.show()

    return policyNet


def visualizeWeights(model):
    for param in model.parameters():
        np_filters = param.cpu().detach().numpy()
        for filter in np_filters:
            plt.imshow(np.squeeze(filter), cmap='gray')
            plt.show()
        break
            

if __name__ == "__main__":
    E_start = 0.9  # Initial probability to choose random action
    E_end = 0.1  # Final probability to choose random action
    model = trainLoop(None, 100, E_start)
    model = trainLoop(model, 100, 0.5)
    model = trainLoop(model, 100, E_end)
    visualizeWeights(model)
    torch.save(model, "models/savedModels/firstModel.pt")


