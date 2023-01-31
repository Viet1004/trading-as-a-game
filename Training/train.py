
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
    action = torch.tensor(0)
    portfolio = INIT_CASH
    init_time = 9
    state = torch.unsqueeze(state_vector(init_time, action, data).float(), 0)
    loss_evo = []
    for time in tqdm(np.arange(init_time, steps)):
        ## Select the greedy action w.r.t Q_network
    #    previous_state = state.detach().clone()
        hidden = Q_network.init_hidden()
        output_NN = Q_network(state, hidden)

        previous_action = action.detach().clone()
#        print("estimated Q:", output_NN)
        action = output_NN.data.max(1)
    #    print(action)
        action = action[1].view(1, 1)
#        print("Optimal action: ", action)
        ## Receive reward r
        portfolio = portfolio_value(time=time,action=action, previous_action=previous_action, previous_value=portfolio, exchange=exchange, data=data)
        state = state_vector(time,action,data).unsqueeze(0).float()  ## The next state s', vector (4*8+6,)
        reward_list = reward_ensemble(time, previous_action,portfolio, exchange, data).float()
#        print("Shape of reward_list: ", reward_list.shape)
        state_list = state_vector_ensemble(time, data).float()
#        print("Shape of state_list: ", state_list.shape)
#        print("Shape of state_list: ", state_list.shape)
        memory.push((state, reward_list, state_list))
        if len(memory) == REPLAY_MEM and time%LEARNING_TIMESTEP == 0:
            sars = memory.sample(LEARNING_TIMESTEP)  # sars : State, Action, Reward, Next_state
#            print(sars)
            batch_state, batch_reward, batch_next_state = zip(*sars)

            ## Attention: Adjust the shape of the vector
            batch_state = torch.cat(batch_state)
#            print("Batch state: ", batch_state)
#            print("Shape of batch_state: ", batch_state.shape)
#            print("The size of batch of stage: ", batch_state.shape)
            batch_reward = torch.cat(batch_reward)
#            print("Shape of batch_reward: ", batch_reward.shape)
            batch_reward = batch_reward.view((LEARNING_TIMESTEP,3))
#            print("Batch reward: ", batch_reward)
            batch_next_state = torch.cat(batch_next_state)
#            print("Shape of batch_reward: ", batch_next_state.shape)
            batch_next_state = batch_next_state.view((LEARNING_TIMESTEP,3,38))
#            print("batch_next_state: ", batch_next_state)
#            print("The size of batch of next_stage: ", batch_next_state.shape)
            
            ## Testing
#            hidden = Q_network.init_hidden(seq_length = LEARNING_TIMESTEP)
#            Q_value_next_state = Q_network(batch_next_state, hidden)
#            print("The size of output of Q_network: ", Q_value_next_state.shape)
#            optimal_action = Q_value_next_state.data.max(2)[1]
#            hidden = target_network.init_hidden(LEARNING_TIMESTEP)
#            print("Optimal actions: ",optimal_action, optimal_action.dtype)
#            
#            estimated_Q_target_network = target_network(batch_next_state, hidden)[optimal_action]
            ## Test with max target network
            Q_value_next_state = target_network(batch_next_state, hidden)
            estimated_Q_target_network = Q_value_next_state.data.max(2)[0]
            

#            print("Estimated Q target network", estimated_Q_target_network.shape)
            hidden = Q_network.init_hidden(LEARNING_TIMESTEP)
            estimated_Q_network = Q_network(batch_state, hidden)
#            print("Shape of batch_reward: ", batch_reward.shape)
#            print("Shape of Q_target_network: ", estimated_Q_target_network.shape)
#            print("Shape of Q_network: ", estimated_Q_network.shape)
#            print("Estimated Q network: ", estimated_Q_network)
#            print("discount: ", batch_reward + DF*estimated_Q_target_network)
            loss = loss_function(batch_reward + DF*estimated_Q_target_network, estimated_Q_network)
#            print("Loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            print(loss)
            loss_evo.append(loss.data.numpy())

        for target_param, param in zip(target_network.parameters(), Q_network.parameters()):
            target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
    
    torch.save(target_network.state_dict(), Path(f"../models/model_{exchange}.pt"))

    plt.plot(loss_evo)
    plt.show()



def test_function(data, exchange):
    model = FinancialDRQN()
    model =  model.load_state_dict(torch.load(Path(f"../models/model_{exchange}.pt")))
    steps = data[0].shape[0]
    action = torch.tensor(0)
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
#        print("estimated Q:", output_NN)
        action = output_NN.data.max(1)
    #    print(action)
        action = action[1].view(1, 1)
        ## Receive reward r
        portfolio = portfolio_value(time=time,action=action, previous_action=previous_action, previous_value=portfolio, exchange=exchange, data=data)
        reward_signal = reward(time, action, previous_action, portfolio, exchange, data)
        state = state_vector(time,action,data).unsqueeze(0).float()  ## The next state s', vector (4*8+6,)
        portfolio_evo.append(portfolio.data.numpy())
        reward_evo.append(reward_signal.data.numpy())
    
    print(f"The annualized return of the {exchange} is : {sum(reward_evo)*252/22}")
    plt.plot(portfolio_evo)
    plt.title("Portfolio value in time")
    plt.plot(reward_evo)
    plt.title("Reward in time")




