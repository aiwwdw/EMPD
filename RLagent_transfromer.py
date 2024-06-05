import random
from Players import Player,Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
from RLENV import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import os
import math
import pdb 

    
class tranformer(Player):
    def __init__(self, name, num, alpha=0.001, gamma=0.99, epsilon=0.1, history_length=10, input_dim=5, history_dim=10):
        super().__init__(name, num)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.memory = deque(maxlen=100)  # Replay memory
        self.batch_size = 32
        self.update_target_every = 5
        self.step = 0
        self.history = {}
        self.prev_history={}

        self.q_network = QNetwork(input_dim, history_dim)
        self.target_network = QNetwork(input_dim, history_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):
        
        base_list = [1, 0, 0, 0, 0]

        if opponent_player not in self.history:
            self.history[opponent_player] = [base_list] * self.history_length
            self.prev_history[opponent_player] = [base_list] * self.history_length

        self.prev_history[opponent_player] = self.history[opponent_player]
        
        self.history[opponent_player].append(current_state)

        if len(self.history[opponent_player]) > self.history_length:
            self.history[opponent_player].pop(0)
        
        if (agent_last_action,opponent_last_action) == ("Cooperate","Cooperate"):
            current_state = [0,0,0,0,1]
        elif (agent_last_action,opponent_last_action) == ("Cooperate","Betray"):
            current_state = [0,0,0,1,0]
        elif (agent_last_action,opponent_last_action) == ("Betray","Cooperate"):
            current_state = [0,0,1,0,0]
        elif (agent_last_action,opponent_last_action) == ("Betray","Betray"):
            current_state = [0,1,0,0,0]
        
       

        if np.random.rand() <= self.epsilon:
            return random.choice(["Cooperate", "Betray"])
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        action =  "Cooperate" if q_values[0][0] > q_values[0][1] else "Betray"
        return action

    def update_q_table(self, state, action, reward, next_state):
        
        self.memory.append((state, action, reward, next_state)) # remember
        
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)

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
