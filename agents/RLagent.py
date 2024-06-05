import random
from Players import Player,Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
# from RLENV import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import os
import copy
import math
import pdb 
import sys


class QNetwork(nn.Module):
    def __init__(self, input_dim, history_num):
        super(QNetwork, self).__init__()
        self.history_num = history_num
        
        # Linear layer to expand input_dim to 64
        self.fc1 = nn.Linear(input_dim, 64)
        
        # Convolutional layers to convert (batch_size, 1, history_num, 64) to (batch_size, 512, 1, 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 10), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 10), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 10), stride=2, padding=1)
        
        # Fully Connected layer to convert 512 to 2
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # x shape: (batch_size, history_num, input_dim)
        batch_size = x.size(0)
        # Apply linear layer to each input in the history
        x = self.fc1(x)
        
        # Reshape to (batch_size, 1, history_num, 256)
        x = x.unsqueeze(1)

        # Apply Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply global average pooling to reduce (batch_size, 512, history_num, 256) to (batch_size, 512, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten and apply final FC layer
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        
        return x

    

class DQN(Player):
    def __init__(self, name, num, alpha=0.001, gamma=0.99, epsilon=0.1, history_length=10, input_dim=5, history_dim=10, output_path='./results'):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.memory = deque(maxlen=5000)  # Replay memory
        self.batch_size = 128
        self.update_target_every = 40
        self.step = 0
        self.history = {}
        self.prev_history={}

        self.q_network = QNetwork(input_dim, history_dim).to('cuda')
        self.target_network = QNetwork(input_dim, history_dim).to('cuda')
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):
        
        base_list = [1, 0, 0, 0, 0]

        if opponent_player not in self.history:
            self.history[opponent_player] = [base_list] * self.history_length
            self.prev_history[opponent_player] = [base_list] * self.history_length

        self.prev_history[opponent_player] = copy.deepcopy(self.history[opponent_player])

        if (agent_last_action,opponent_last_action) == ("Cooperate","Cooperate"):
            action_pair = [0,0,0,0,1]
        elif (agent_last_action,opponent_last_action) == ("Cooperate","Betray"):
            action_pair = [0,0,0,1,0]
        elif (agent_last_action,opponent_last_action) == ("Betray","Cooperate"):
            action_pair = [0,0,1,0,0]
        elif (agent_last_action,opponent_last_action) == ("Betray","Betray"):
            action_pair = [0,1,0,0,0]
        else: 
            print("Wrong last action tuples")
            sys.exit(1)
        
        self.history[opponent_player].append(action_pair)
        self.history[opponent_player].pop(0)
        

        if np.random.rand() <= self.epsilon:
            return random.choice(["Cooperate", "Betray"])
        
        state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')
        with torch.no_grad():
            q_values = self.q_network(state)
        action =  "Cooperate" if q_values[0][0] < q_values[0][1] else "Betray"
        return action
    

    def update_q_table(self, action, reward, opponent_player):
        
        if action == "Cooperate":
            action = 1
        elif action == "Betray":
            action = 0
            
        self.memory.append((self.prev_history[opponent_player], action, reward, self.history[opponent_player])) # remember
        
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch).to('cuda')
        action_batch = torch.LongTensor(action_batch).to('cuda')
        reward_batch = torch.FloatTensor(reward_batch).to('cuda')
        next_state_batch = torch.FloatTensor(next_state_batch).to('cuda')

        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        max_next_q_values = self.target_network(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (self.gamma * max_next_q_values)

        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.step += 1


class Q_learning_business(Player):
    def __init__(self, name, num, alpha=0.1, gamma=0.9, epsilon=0.1, history_length=3, output_path='./results'):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.q_table = {}  # Q-table to store action-values
        self.last_action = None
        self.output_path = output_path
        self.count = 0 

        self.past_history = {}
        self.current_history = {}
        self.temp_history = {}


    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):

        self.load_q_table(os.path.join(self.output_path, "q_table.pkl"))

        if opponent_player not in self.past_history:
            self.past_history[opponent_player]= []
            self.current_history[opponent_player] = []
        if opponent_player not in self.current_history:
            self.current_history[opponent_player] = []

        self.current_history[opponent_player].append((agent_last_action,opponent_last_action, True))
        
        if len(self.current_history[opponent_player]) == self.history_length:
            self.temp_history[opponent_player] = [(x, y, not z) for x, y, z in self.current_history[opponent_player]]
        
        if len(self.current_history[opponent_player]) > self.history_length:
            self.current_history[opponent_player].pop(0)

        state = tuple(self.current_history[opponent_player] + self.past_history[opponent_player])
        if random.random() < epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.last_action = action
        self.count += 1
        return action


    def update_q_table(self, reward, opponent_player):
        
        state = tuple(self.current_history[opponent_player] + self.past_history[opponent_player])

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        prev_q_value = self.q_table[state][self.last_action]
        max_q_value = max(self.q_table[state].values())
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
        self.q_table[state][self.last_action] = new_q_value

        self.save_q_table(os.path.join(self.output_path, "q_table.pkl"))


    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


class Q_learning(Player):
    def __init__(self, name, num, alpha=0.1, gamma=0.9, epsilon=0.1, history_length=4, output_path='./results'):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.q_table = {}  # Q-table to store action-values
        self.last_action = None
        self.history = {}
        self.output_path = output_path
        self.count = 0 


    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):

        self.load_q_table(os.path.join(self.output_path, "q_table.pkl"))

        if opponent_player not in self.history:
            self.history[opponent_player] = []

        self.history[opponent_player].append((agent_last_action,opponent_last_action))
        
        if len(self.history[opponent_player]) > self.history_length:
            self.history[opponent_player].pop(0)

        state = tuple(self.history[opponent_player])

        if random.random() < epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.last_action = action
        self.count += 1
        return action


    def update_q_table(self, reward, opponent_player):
        
        state = tuple(self.history[opponent_player])

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        prev_q_value = self.q_table[state][self.last_action]
        max_q_value = max(self.q_table[state].values())
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
        self.q_table[state][self.last_action] = new_q_value

        self.save_q_table(os.path.join(self.output_path, "q_table.pkl"))


    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

"""
RLPLayer 와 Smarty의 차이점은
learning parameter의 차이점이 있다. 
"""

class RLPlayer(Player):
    def __init__(self, name, num, alpha=0.1, gamma=0.9, epsilon=0.1, history_length=2, output_path='./results'):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.q_table = {}  # Q-table to store action-values
        self.opponent_history = []  # List to store opponent action history
        self.last_action = None
        self.output_path = output_path


    def perform_action(self, agent_last_action ,opponent_last_action, round_number,opponent_player):

        self.load_q_table(os.path.join(self.output_path, "simple_q_table.pkl"))

  
        self.opponent_history.append(opponent_last_action)
        if len(self.opponent_history) > self.history_length:
            self.opponent_history.pop(0)  

        state = tuple(self.opponent_history)

        if random.random() < self.epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_action = action
        return action

    def update_q_table(self, reward, opponent_player):

        state = tuple(self.opponent_history)

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        prev_q_value = self.q_table[state][self.last_action]
        max_q_value = max(self.q_table[state].values())
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
        self.q_table[state][self.last_action] = new_q_value
        self.save_q_table(os.path.join(self.output_path, "simple_q_table.pkl"))


    def reset(self):
        self.opponent_history = []
        self.last_action = None

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

class Smarty(Player):
    def __init__(self, name, num, alpha=0.2, gamma=0.95, epsilon=0.08, history_length=4, output_path='./results'):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.q_table = {}  # Q-table to store action-values
        self.opponent_history = []  # List to store opponent action history
        self.last_action = None
        self.output_path = output_path


    def perform_action(self, agent_last_action ,opponent_last_action, round_number,opponent_player):

        self.load_q_table(os.path.join(self.output_path, "smarty_table.pkl"))

  
        self.opponent_history.append(opponent_last_action)
        if len(self.opponent_history) > self.history_length:
            self.opponent_history.pop(0)  

        state = tuple(self.opponent_history)

        if random.random() < self.epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_action = action
        return action

    def update_q_table(self, reward, opponent_player):

        state = tuple(self.opponent_history)

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        prev_q_value = self.q_table[state][self.last_action]
        max_q_value = max(self.q_table[state].values())
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
        self.q_table[state][self.last_action] = new_q_value
        self.save_q_table(os.path.join(self.output_path, "smarty_table.pkl"))


    def reset(self):
        self.opponent_history = []
        self.last_action = None

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
