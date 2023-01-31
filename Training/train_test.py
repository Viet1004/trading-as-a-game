
import datetime as dt
import numpy as np
from model import FinancialDRQN, ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from utils import reward, state_vector, reward_ensemble, state_vector_ensemble, portfolio_value
from tqdm import tqdm
import pickle
from pathlib import Path

torch.manual_seed(0)

LEARNING_TIMESTEP = 64
REPLAY_MEM = 480
LR = 0.00025 
DF = 0.99 # Discount factor
TAU = 0.001 # Target network
EPOCHS = 10000
N_EP = 2000
INIT_CASH = 100000
N_CURRENCY = 4
EPSILON = 0.03

exchange_dict = {"AUDNZD": 0, "AUDUSD": 1, "CADJPY": 2, "CHFJPY": 3}

def train_function(data, exchange):
    Q_network = FinancialDRQN()
    model_parameter = Q_network.state_dict() 
    target_network = FinancialDRQN()
    target_network.load_state_dict(model_parameter)
    optimizer = optim.Adam(Q_network.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    memory = ReplayMemory(REPLAY_MEM)
    steps = data[0].shape[0]
    action = torch.tensor(1)
    portfolio = INIT_CASH
    init_time = 9
    state = torch.unsqueeze(state_vector(init_time, action, data).float(), 0)
    loss_evo = []
    for time in tqdm(np.arange(init_time, steps)):
        ## Select the greedy action w.r.t Q_network
        hidden = Q_network.init_hidden()

        previous_action = action.detach().clone()
        if np.random.uniform() < EPSILON:
            action = np.random.choice([-1,0,1])
            action = torch.tensor(action).view(1,1)
        else:
            output_NN = Q_network(state, hidden)
            action = output_NN.data.max(1)
            action = (action[1]-1).view(1, 1)
        ## Value of the portfolio at the moment
        portfolio = portfolio_value(time=time,action=action, previous_action=previous_action, previous_value=portfolio, exchange=exchange, data=data)
        state = state_vector(time,action,data).unsqueeze(0).float()  ## The next state s', vector (4*8+6,)
        reward_list = reward_ensemble(time, previous_action,portfolio, exchange, data).float()
        state_list = state_vector_ensemble(time, data).float()
        memory.push((state, reward_list, state_list))   ## Save the chain State, Action, Reward, Next_state in the memory
        if len(memory) == REPLAY_MEM and time%LEARNING_TIMESTEP == 0:
            sars = memory.sample(LEARNING_TIMESTEP)  # sars : State, Action, Reward, Next_state
            batch_state, batch_reward, batch_next_state = zip(*sars)

            batch_state = torch.cat(batch_state)
            batch_reward = torch.cat(batch_reward)
            batch_reward = batch_reward.view((LEARNING_TIMESTEP,3))
            batch_next_state = torch.cat(batch_next_state)
            batch_next_state = batch_next_state.view((LEARNING_TIMESTEP,3,38))
            Q_value_next_state = target_network(batch_next_state, hidden)
            estimated_Q_target_network = Q_value_next_state.data.max(2)[0]
            

            hidden = Q_network.init_hidden(LEARNING_TIMESTEP)
            estimated_Q_network = Q_network(batch_state, hidden)
            loss = loss_function(batch_reward + DF*estimated_Q_target_network, estimated_Q_network)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_evo.append(loss.data.numpy())

        for target_param, param in zip(target_network.parameters(), Q_network.parameters()):
            target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
    

    plt.plot(loss_evo)
    plt.show()

    return target_network



def test_function(model, data, exchange):
    steps = data[0].shape[0]
    action = torch.tensor(1)
    portfolio = INIT_CASH
    init_time = 9
    state = torch.unsqueeze(state_vector(init_time, action, data).float(), 0)
    portfolio_evo = []
    reward_evo = []
    for time in tqdm(np.arange(init_time, steps)):
        ## Select the greedy action w.r.t Q_network
        hidden = model.init_hidden()
        output_NN = model(state, hidden)

        previous_action = action.detach().clone()
        action = output_NN.data.max(1)
        action = (action[1]-1).view(1, 1)

        portfolio = portfolio_value(time=time,action=action, previous_action=previous_action, previous_value=portfolio, exchange=exchange, data=data)
        reward_signal = reward(time, action, previous_action, portfolio, exchange, data)
        state = state_vector(time,action,data).unsqueeze(0).float()  ## The next state s', vector (4*8+6,)
        portfolio_evo.append(portfolio.view(1).data.numpy())
        reward_evo.append(reward_signal.view(1).data.numpy())
    
    print(f"The annualized return of the {exchange} is : {sum(reward_evo)*252/22}")
    plt.figure()
    plt.plot(portfolio_evo)
    plt.title("Portfolio value in time")
    plt.show()
    plt.figure()
    plt.plot(reward_evo)
    plt.title("Reward in time")
    plt.show()




