# Train the Dqn model to play 4 in a row

from aiPlayer import aiPlayer
from fourinarowGame import FourinarowGame
from models.dqn import Dqn
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import copy


# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def trainLoop(policyNet=None, episodes=10, E=0.5):
    G = 0.9  # Discount factor
    LR = 1e-2  # Learning rate

    if policyNet == None:
        policyNet = Dqn()  # Updated after all episodes, and saved (?) Moves are made based on this.
    policyNet.to(device)
    targetNet = Dqn().to(device)  # Updated after every episode, based on policyNet (?)
    # Load the policyNet state to targetNet
    targetNet.load_state_dict(policyNet.state_dict())

    p1 = aiPlayer("P1", policyNet)
    p2 = aiPlayer("P2", policyNet)

    optimizer = torch.optim.AdamW(policyNet.parameters(), lr=LR)
    lossFunction = torch.nn.MSELoss()
    train_losses = []

    state_memory = []
    action_memory = []
    new_state_memory = []
    reward_memory = []

    # Play the game for a certain amount of episodes (1 episode = 1 game) and
    # save the states, actions and rewards for training
    for ep in range(episodes):
        sa_memory = []  # For saving states and actions
        game = FourinarowGame(p1, p2)
        temp_losses = []
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

            state_memory.append(state)
            action_memory.append(action)
            new_state_memory.append(new_state)
            reward_memory.append(reward)


            if game.getInturn().get_model_name() == "random":
                # Do not try to train random model
                continue
                    
                    
        # Optimize model (in epochs)
        #epochs = 10
        #batch_size = 1
        #for epoch in range(epochs):
            #batch_i = random.sample(range(len(state_memory)), batch_size)
            #b_states = torch.from_numpy(np.array(state_memory)[batch_i]).float().to(device)
            #b_actions = torch.from_numpy(np.array(action_memory)[batch_i]).to(device)
            #b_new_states = torch.from_numpy(np.array(new_state_memory)[batch_i]).float().to(device)
            #b_rewards = torch.from_numpy(np.array(reward_memory)[batch_i]).to(device)

            # Optimize model

            b_states = torch.tensor(state).float().to(device)
            b_actions = torch.tensor(action).to(device)
            b_new_states = torch.tensor(new_state).float().to(device)
            b_rewards = torch.tensor(reward).to(device)

            # Prepare samples for DQN
            b_states = b_states[None, :]
            b_new_states = b_new_states[None, :]

            # Q-value for the current state
            Q_pred = policyNet(b_states)
            # New Q calculated with r and max Q at next state
            # (Target net: "ground truth")
            Q_target = 1 * Q_pred
            Q_target[b_actions] = b_rewards + G * torch.max(targetNet(b_new_states))

            loss = lossFunction(Q_pred, Q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_losses.append(loss.cpu().detach().numpy())
        train_losses.append(np.mean(temp_losses))

    return policyNet, train_losses


def visualizeWeights(model):
    for param in model.parameters():
        np_filters = param.cpu().detach().numpy()
        for filter in np_filters:
            plt.imshow(np.squeeze(filter), cmap='gray')
            plt.show()
        return np_filters
            

if __name__ == "__main__":
    E_start = 0.9  # Initial probability to choose random action
    E_end = 0.1  # Final probability to choose random action
    epsilon = np.arange(E_start, E_end, -0.1)

    model = Dqn()
    model.to(device)

    # Train with decreasing epsilon and save the losses for visualization
    all_losses = []
    episodes = 5000
    for e in epsilon:
        print(e)
        model, step_losses = trainLoop(model, episodes, e)
        all_losses += step_losses
    
    model, step_losses = trainLoop(model, episodes, E_end)
    all_losses += step_losses

    torch.save(model, "models/savedModels/9step_5000ep_model.pt")

    # Visualize losses and weights
    plt.plot(all_losses)
    plt.show()
    #end_filters = visualizeWeights(model)
