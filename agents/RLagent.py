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
        print(state)
        
        if random.random() < self.epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_action = action
        return action

    def update_q_table(self, reward, opponent_player, player1_last_action, player2_last_action):

        state = tuple(self.opponent_history)

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 1, "Betray": 0}

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

    def update_q_table(self, reward, opponent_player, player1_last_action, player2_last_action):

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
