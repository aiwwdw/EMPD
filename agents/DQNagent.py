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
        self.fc1 = nn.Linear(input_dim, 16)
        
        # Convolutional layers to convert (batch_size, 1, history_num, 16) to (batch_size, 16, 1, 1)
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=1)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2), stride=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2), stride=1)
        
        # Size = 2
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 5))
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5))
        # self.bn2 = nn.BatchNorm2d(16)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5))
        # self.bn3 = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        
        # Fully Connected layer to convert 512 to ㅌㄷ2
        # self.fc2 = nn.Linear(16, 2)
        self.fc2 = nn.Linear(256*5, 2)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # x shape: (batch_size, history_num, input_dim)
        batch_size = x.size(0)
        # Apply linear layer to each input in the history
        x = self.fc1(x)
        x = F.relu(x)
        
        # Reshape to (batch_size, 1, history_num, 256)
        x = x.unsqueeze(1)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply global average pooling to reduce (batch_size, 512, history_num, 256) to (batch_size, 512, 1, 1)
        # x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten and apply final FC layer
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        
        return x


# class QNetwork(nn.Module):
#     def __init__(self, input_dim, history_num):
#         super(QNetwork, self).__init__()
#         self.history_num = history_num
        
#         # Linear layer to expand input_dim to 64
#         self.fc1 = nn.Linear(input_dim, 64)
        
#         # Convolutional layers to convert (batch_size, 1, history_num, 64) to (batch_size, 512, 1, 1)
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 10), stride=2, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 10), stride=2, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 10), stride=2, padding=1)
        
#         # Fully Connected layer to convert 512 to 
#         self.fc2 = nn.Linear(16, 2)
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#     def forward(self, x):
#         # x shape: (batch_size, history_num, input_dim)
#         batch_size = x.size(0)
#         # Apply linear layer to each input in the history
#         x = self.fc1(x)
#         # x = F.relu(x)
        
#         # Reshape to (batch_size, 1, history_num, 256)
#         x = x.unsqueeze(1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         # Apply global average pooling to reduce (batch_size, 512, history_num, 256) to (batch_size, 512, 1, 1)
#         x = F.adaptive_avg_pool2d(x, (1, 1))

#         # Flatten and apply final FC layer
#         x = x.view(batch_size, -1)
#         x = self.fc2(x)
        
#         return x

class DQN(Player):
    def __init__(self, 
                 name, 
                 num, 
                 alpha=0.001,
                 gamma=0.99,
                 history_length = 3,
                 batch_size = 32,
                 input_dim= 3,
                 update_target_every = 10, #100
                 output_path='./results'
                 ):
        
        super().__init__(name, num)
        ########################################################################
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.history_length = history_length  # Length of opponent action history
        self.memory = deque(maxlen=1000)  # Replay memory
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.update_target_every = update_target_every
        self.output_path = output_path
        ########################################################################

        self.q_network = QNetwork(self.input_dim, self.history_length).to('cuda')
        self.target_network = QNetwork(self.input_dim, self.history_length).to('cuda')
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        
        self.epsilon = None  # Exploration rate
        self.step = 0
        self.history = {}
        self.prev_history={}
        self.loss_fn = nn.MSELoss()

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):
        self.epsilon = epsilon
        
        base_list = [1, 0, 0]
        if opponent_player not in self.history:
            self.history[opponent_player] = [base_list] * self.history_length
            self.prev_history[opponent_player] = [base_list] * self.history_length

        if np.random.rand() <= self.epsilon:
            choice = random.choice(["Cooperate", "Betray"])
            print(choice)
            return choice

        state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')
        with torch.no_grad():
            # self.q_network.eval()
            q_values = self.q_network(state)
            # self.q_network.train()
    
        print(q_values)
        action =  "Cooperate" if q_values[0][0] < q_values[0][1] else "Betray"
        return action
    
    
    def update_q_table(self, reward, opponent_player, agent_action, opponent_action,mode, done):

        self.prev_history[opponent_player] = copy.deepcopy(self.history[opponent_player])

        # if (agent_last_action,opponent_last_action) == ("Cooperate","Cooperate"):
        #     action_pair = [0,0,0,0,1]
        # elif (agent_last_action,opponent_last_action) == ("Cooperate","Betray"):
        #     action_pair = [0,0,0,1,0]
        # elif (agent_last_action,opponent_last_action) == ("Betray","Cooperate"):
        #     action_pair = [0,0,1,0,0]
        # elif (agent_last_action,opponent_last_action) == ("Betray","Betray"):
        #     action_pair = [0,1,0,0,0]
        # else: 
        #     print("Wrong last action tuples")
        #     sys.exit(1)
        if (agent_action,opponent_action) == ("Cooperate","Cooperate"):
            action_pair = [0,1,1]
        elif (agent_action,opponent_action) == ("Cooperate","Betray"):
            action_pair = [0,1,0]
        elif (agent_action,opponent_action) == ("Betray","Cooperate"):
            action_pair = [0,0,1]
        elif (agent_action,opponent_action) == ("Betray","Betray"):
            action_pair = [0,0,0]
        else: 
            print("Wrong last action tuples")
            sys.exit(1)

        self.history[opponent_player].append(action_pair)
        self.history[opponent_player].pop(0)
        
        if agent_action == "Cooperate":
            action = 1
        elif agent_action == "Betray":
            action = 0

        if mode == 'train':
            self.memory.append((self.prev_history[opponent_player], action, reward, self.history[opponent_player], done))
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.FloatTensor(state_batch).to('cuda')
            action_batch = torch.LongTensor(action_batch).to('cuda')
            reward_batch = torch.FloatTensor(reward_batch).to('cuda')
            next_state_batch = torch.FloatTensor(next_state_batch).to('cuda')
            done_batch = torch.FloatTensor(done_batch).to('cuda')


            current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            max_next_q_values = self.target_network(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + (self.gamma * max_next_q_values * (1-done_batch))
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step += 1
            if self.step % self.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()
    
    def save_model(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer' : self.memory,
            'step': self.step,
        }, os.path.join(path,'checkpoints.pt'))
        # print(f"Model saved to {path}.pt")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['replay_buffer']
        self.step = checkpoint['step']
        # print(f"Model loaded from {path}")
    
    def check_network(self, opponent_player, history, epsilon):
        self.history = history
        self.epsilon = epsilon
        
        base_list = [1, 0, 0]
        if opponent_player not in self.history:
            self.history[opponent_player] = [base_list] * self.history_length
            self.prev_history[opponent_player] = [base_list] * self.history_length

        if np.random.rand() <= self.epsilon:
            return random.choice(["Cooperate", "Betray"])

        state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values[0][0], q_values[0][1]


