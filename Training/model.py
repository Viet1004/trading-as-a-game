import random
import math
import numpy as np
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

N_PAIRS = 6
N_HIDDEN_UNITS = 128
INPUTS_DIM = 38
N_HIDDEN_UNITS_LSTM = 256

class FinancialDRQN(nn.Module):
    def __init__(self) -> None:
        super(FinancialDRQN, self).__init__()
        ## Two first linear layers
        net1 = nn.Linear(INPUTS_DIM, N_HIDDEN_UNITS)
        nn.init.eye_(net1.weight)
        layer1 = [net1, nn.ELU()]
        self.net1 = nn.Sequential(*layer1)
        
        net2 = nn.Linear(N_HIDDEN_UNITS, N_HIDDEN_UNITS)
        nn.init.eye_(net2.weight)
        layer2 = [net2, nn.ELU()]
        self.net2 = nn.Sequential(*layer2)
        ## LTSM layer
        self.lstm = nn.LSTM( N_HIDDEN_UNITS, N_HIDDEN_UNITS_LSTM, 1, batch_first=True)
        ## Last linear layer
        net4 = nn.Linear(N_HIDDEN_UNITS_LSTM, 3)
        nn.init.normal_(net4.weight, std=0.001)
        self.net4 = net4

    def init_hidden(self, seq_length = 1, batch_size = 1):
        return torch.zeros( seq_length, batch_size, N_HIDDEN_UNITS_LSTM)

    def forward(self, x, hidden):
        x = self.net1(x)
        x = self.net2(x)
        x, h_0 = self.lstm(x)
        x = self.net4(x)
        return x.float()

class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    
    def sample(self, seq_length):
        start = np.random.randint(0, len(self.memory) - seq_length)
        return self.memory[start:start+seq_length]

    def __len__(self):
        return len(self.memory)



