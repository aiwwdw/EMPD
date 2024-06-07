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


class LSTMQNetwork(nn.Module):
    def __init__(self, input_dim, history_num):
        super(LSTMQNetwork, self).__init__()
        self.hidden_input_space = 64
        self.hidden_output_space = 1024
        self.history_num = history_num
        
        # Linear layer to expand input_dim to 64 
        self.Linear1 = nn.Linear(input_dim, self.hidden_input_space)
        self.lstm    = nn.LSTM(self.hidden_input_space,self.hidden_output_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_output_space, self.hidden_input_space)
        
        # Fully Connected layer to convert 512 to 2
        self.fc2 = nn.Linear(self.hidden_input_space, 2)

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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.Linear1(x))
        x,_ = self.lstm(x)
        x = F.relu(x)
        x = x[:, -1, :].unsqueeze(1)
        x = F.relu(self.Linear2(x))
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        return x
    
    
class ConvLstmQNetwork(nn.Module):
    def __init__(self, input_dim, history_num):
        super(ConvLstmQNetwork, self).__init__()
        self.history_num = history_num
        
        # self.fc1 = nn.Linear(input_dim, 64)
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 10), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 10), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 10), stride=2, padding=1)
        
        # Fully Connected layer to convert 512 to 2
        self.fc2 = nn.Linear(512, 2)
    
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Apply linear layer to each input in the history
        
        x,_ = self.lstm(x)
        x = F.relu(x)
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

    

class LSTMQN(Player):
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
        self.output_path = output_path

        self.q_network = ConvLstmQNetwork(input_dim, history_dim).to('cuda')
        self.target_network = ConvLstmQNetwork(input_dim, history_dim).to('cuda')
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.train() # eval()
        
    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):
        self.epsilon = epsilon
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
            # print("Exploration")
            return random.choice(["Cooperate", "Betray"])
        # else:
        #     print("Exploitation")

        state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')
        self.q_network.train()
        with torch.no_grad():
            q_values = self.q_network(state,)
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
            # self.save_model(path = self.output_path)
        
        self.step += 1

    
    def save_model(self, path):
        torch.save({
            'lstm_q_network_state_dict': self.q_network.state_dict(),
            'lstm_target_network_state_dict': self.target_network.state_dict(),
            'lstm_optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path,'checkpoints.pt'))
        print(f"Model saved to {path}.pt")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['lstm_q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['lstm_target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['lstm_optimizer_state_dict'])
        print(f"Model loaded from {path}")


