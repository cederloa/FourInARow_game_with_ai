# Train the Dqn model to play 4 in a row

from aiPlayer import aiPlayer
from fourinarowGame import FourinarowGame
from models.dqn import Dqn
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import time


# Global variables
G = 0.9  # Discount factor
LR = 1e-3  # Learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossFunction = torch.nn.MSELoss()


def makeAMove(player, game, E):
    '''
    Play one move as given player.

    return:
        game: updated game
        state: state of the game before making the move
        action: the action/move made by the player
        new_state: state of the game after making the move, in opposing player's
                point of view
    '''
    # Get the current player's state of the game
    player.set_state(game.getGameBoard())
    state = player.get_state()

    # Choose action and play it on the gameboard
    isRandom = random.uniform(0, 1) < E
    action = player.choose_action(isRandom)
    game.playTurn(action)

    # Get the current player's new state of the game, and turn it to opposing
    # player's point of view.
    player.set_state(game.getGameBoard())
    new_state = -1 * player.get_state()

    return game, state, action, new_state


def optimize(state, action, new_state, reward, policyNet, optimizer,
             targetNet=None):
    '''
    One step of optimization for the policyNet given current state, action, new
    state and reward.
    
    return: loss, for visualization
    '''
    if targetNet == None:
        targetNet = policyNet
    
    # Prepare samples for DQN
    b_states = torch.tensor(state).float().to(device)
    b_actions = torch.tensor(action).to(device)
    b_new_states = torch.tensor(new_state).float().to(device)
    b_rewards = torch.tensor(reward).to(device)

    b_states = b_states[None, :]
    b_new_states = b_new_states[None, :]

    # Q-value for the current state
    Q_pred = policyNet(b_states)

    # Target Q given by r or max Q at next state
    # (Target net: "ground truth")
    Q_target = 1 * Q_pred
    Q_target[b_actions] = (G * (-1 * torch.max(targetNet(b_new_states)))
                           if b_rewards == 0
                           else b_rewards)

    # Optimize based on the loss
    loss = lossFunction(Q_pred, Q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.cpu().detach().numpy()


def visualizeWeights(model):
    '''
    Visualizing model weights. Used in debugging/refining the model.
    '''
    for param in model.parameters():
        np_filters = param.cpu().detach().numpy()
        for filter in np_filters:
            plt.imshow(np.squeeze(filter), cmap='gray')
            plt.show()
        return np_filters


def trainLoop(policyNet1=None, policyNet2=None, episodes=10, E=0.5):
    '''
    Loop for training the RL models. Plays a number of games given by parameter
    'episodes'. Parameter 'E' defines the probability to choose a random action.

    return:
        policyNet: the trained RL model
        train_losses: training losses for visualizing model performance
    '''
    # policyNet is updated after each move. Moves are made based on it.
    if policyNet1 == None:
        policyNet1 = Dqn()
    
    if policyNet2 == None:
        policyNet2 = Dqn()
    
    policyNet1.to(device)
    policyNet2.to(device)
    
    # targetNet is updated once every training loop, based on policyNet's state
    #targetNet = Dqn().to(device)
    # Load the policyNet state to targetNet
    #targetNet.load_state_dict(policyNet1.state_dict())

    optimizer1 = torch.optim.AdamW(policyNet1.parameters(), lr=LR)
    optimizer2 = torch.optim.AdamW(policyNet2.parameters(), lr=LR)
    train_losses = []

    prev_state = None
    prev_action = None

    # Play the game for a certain amount of episodes (1 episode = 1 game) and
    # save the states, actions and rewards for training
    for ep in range(episodes):
        p1 = aiPlayer("P1", policyNet1)
        p2 = aiPlayer("P2", policyNet1)
        game = FourinarowGame(p1, p2)
        temp_losses = []
        # Play the game until it ends
        while game.getResults() == False:
            if game.getInturn() == p1:
                optimizer = optimizer1
                policyNet = policyNet1
                other_opt = optimizer2
                other_pol = policyNet2
                game, state, action, new_state = makeAMove(p1, game, E)
            else:
                optimizer = optimizer2
                policyNet = policyNet2
                other_opt = optimizer1
                other_pol = policyNet1
                game, state, action, new_state = makeAMove(p2, game, E)

            if game.getResults() == False or game.getResults() == "Draw":
                reward = 0
            else:
                reward = 1
                # Give the loser a -1 reward, and train with it
                temp_loss = optimize(prev_state, prev_action, state, -1,
                                     other_pol, other_opt, policyNet)
                temp_losses.append(temp_loss)

            if game.getInturn().get_model_name() == "random":
                # Do not try to train random model
                continue
            
            # Optimize on each turn played
            temp_loss = optimize(state, action, new_state, reward,
                                 policyNet, optimizer, other_pol)
            temp_losses.append(temp_loss)

            prev_state = state
            prev_action = action

        # Save mean of losses for current episode
        train_losses.append(np.mean(temp_losses))

    return policyNet, train_losses
            

if __name__ == "__main__":

    # Modifiable parameters for training the model
    E_start = 0.1  # Initial probability to choose random action
    E_end = 0.01  # Final probability to choose random action
    steps = 2
    epsilon = np.linspace(E_start, E_end, steps)

    # You can load existing models to train them further
    #p1_model = torch.load("models/savedModels/dualCnn/2step_1000ep_model_p1_noAct.pt")
    #p2_model = torch.load("models/savedModels/dualCnn/2step_1000ep_model_p2_noAct.pt")

    # Start training models from scratch
    p1_model = Dqn()
    p2_model = Dqn()
    p1_model.to(device)
    p2_model.to(device)

    # Train with decreasing epsilon and save the losses for visualization
    all_losses = []
    episodes = 1000
    for e in epsilon:
        print(e)
        model, step_losses = trainLoop(p1_model, p2_model, episodes, e)
        all_losses += step_losses

    torch.save(p1_model,
    f"models/savedModels/dualCnn/{steps}step_{episodes}ep_model_p1_noAct.pt")
    torch.save(p2_model,
    f"models/savedModels/dualCnn/{steps}step_{episodes}ep_model_p2_noAct.pt")

    # Visualize losses (and weights)
    plt.plot(all_losses)
    plt.show()
    #end_filters = visualizeWeights(model)
