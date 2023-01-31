import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import warnings
from sklearn import preprocessing

VOLUME = 10000
SPREAD = 0.0001 # Need to change to see the effect of the spread
NUM_CURRENCY = 4

exchange_dict = {"AUDNZD": 0, "AUDUSD": 1, "CADJPY": 2, "CHFJPY": 3}

def time_to_index(data, time):
    index = data.index[data[time] == time].tolist()[0]
    return index

def state_vector(time, action, data):
    """

    action: takes the value -1,0,1
    """
    dt_object = data[0].iloc[time]["time"]
    dt_object = dt.datetime.strptime( dt_object, "%Y-%m-%d %H:%M:%S")
    week_day = dt_object.weekday()
    hour = dt_object.hour
    minute = dt_object.minute
    time_feature = np.array([np.sin(np.pi*minute/60), np.sin(np.pi*hour/60), np.sin(np.pi*week_day/5)])

    pair_num = len(data)
    market_feature = np.zeros((pair_num,8))
    
    for i in range(pair_num):
        currency = data[i]
#        index = time_to_index(data[i], dt_object)
        historical_data = np.log(data[i].iloc[time-9:time]["close"]).diff().iloc[1:]
        market_feature[i] = np.array(historical_data)
    market_feature = market_feature.flatten()
    market_feature = preprocessing.normalize([market_feature]).flatten()

    onehot_action = [0,1,0]
    if action == -1:
        onehot_action = [1,0,0]
    elif action == 0:
        onehot_action = [0,1,0]
    elif action == 1:
        onehot_action = [0,0,1]
    else:
        warnings.warn(f"The action must be -1,0 or 1, get {action}", UserWarning)
    onehot_action = np.array(onehot_action)

    return torch.from_numpy(np.concatenate((time_feature, market_feature, onehot_action)))

def state_vector_ensemble(time, data):
    state_vector_list = []
    for action in [-1,0,1]:
        state_vector_list.append(state_vector(time, action, data))
    return torch.stack(state_vector_list)

def reward(time, action, previous_action, previous_value, exchange, data):
    index = exchange_dict[exchange]
    d_t = VOLUME * np.abs(action - previous_action) * SPREAD
    v_t = previous_value + action * VOLUME * (data[index].iloc[time]["close"] - data[index].iloc[time]["open"]) - d_t
    return torch.log(v_t/previous_value).squeeze_(1)

def portfolio_value(time, action, previous_action, previous_value, exchange, data):
    index = exchange_dict[exchange]
    d_t = VOLUME * np.abs(action - previous_action) * SPREAD
    v_t = previous_value + action * VOLUME * (data[index].iloc[time]["close"] - data[index].iloc[time]["open"]) - d_t
    return v_t

def reward_ensemble(time, previous_action, previous_value, exchange, data):
    reward_list = []
    for action in [-1,0,1]:
        reward_list.append(reward(time, action, previous_action, previous_value, exchange, data))
    return torch.stack(reward_list)




