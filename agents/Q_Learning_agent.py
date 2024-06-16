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


class Q_learning_business(Player):
    def __init__(self, 
                 name,
                 num,
                 alpha=0.1,
                 gamma=0.9, 
                 epsilon=0.6, 
                 history_length=5,
                 output_path='./results'):
        
        super().__init__(name, num)
        
        #####################################################
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.output_path = output_path
        #####################################################
        self.count = 0 
        self.q_table = {}  # Q-table to store action-values
        self.last_action = None
        self.past_history = {}
        self.current_history = {}
        self.temp_history = {}


    def perform_action(self, player1_last_action, player2_last_action, round_number, opponent_player, epsilon):

        self.load_q_table(os.path.join(self.output_path, "q_table.pkl"))

        if opponent_player not in self.past_history:
            self.past_history[opponent_player]= []
            self.current_history[opponent_player] = []
        if opponent_player not in self.current_history:
            self.current_history[opponent_player] = []


        state = tuple(self.current_history[opponent_player] + self.past_history[opponent_player])

        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        if random.random() < epsilon or state not in self.q_table:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.last_action = action
        self.count += 1
        return action


    def update_q_table(self, reward, opponent_player, agent_last_action ,opponent_last_action,mode, done):
        
        prev_state = tuple(self.current_history[opponent_player] + self.past_history[opponent_player])

        self.current_history[opponent_player].append((agent_last_action,opponent_last_action, True))
        if len(self.current_history[opponent_player]) == self.history_length:
            self.temp_history[opponent_player] = [(x, y, not z) for x, y, z in self.current_history[opponent_player]]
        if len(self.current_history[opponent_player]) > self.history_length:
            self.current_history[opponent_player].pop(0)
        
        state = tuple(self.current_history[opponent_player] + self.past_history[opponent_player])
        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}
        if mode == 'train':
            prev_q_value = self.q_table[prev_state][self.last_action]
            max_q_value = max(self.q_table[state].values())
            new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value * (1-done)- prev_q_value)
            self.q_table[prev_state][self.last_action] = new_q_value

            self.save_q_table(os.path.join(self.output_path, "q_table.pkl"))


    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


class Q_learning(Player):
    def __init__(self, 
                 name, 
                 num, 
                 alpha=0.1, 
                 gamma=0.9, 
                 epsilon=0.6, 
                 history_length=2,
                 output_path='./results'):
        
        super().__init__(name, num)

        #####################################################
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.history_length = history_length  # Length of opponent action history
        self.output_path = output_path
        #####################################################
        self.q_table = {}  # Q-table to store action-values
        self.last_action = None
        self.history = {}
        self.count = 0 


    def perform_action(self, agent_last_action ,opponent_last_action, round_number, opponent_player, epsilon):

        self.load_q_table(os.path.join(self.output_path, "q_table.pkl"))
        if opponent_player not in self.history:
            self.history[opponent_player] = []

        state = tuple(self.history[opponent_player])
        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}
        
        if random.random() < epsilon:
            action = random.choice(["Cooperate", "Betray"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        
        self.last_action = action
        self.count += 1
        return action


    def update_q_table(self, reward, opponent_player, agent_action, opponent_action, mode, done):
        
        prev_state = tuple(self.history[opponent_player])

        self.history[opponent_player].append((agent_action,opponent_action))
        if len(self.history[opponent_player]) > self.history_length:
            self.history[opponent_player].pop(0)

        state = tuple(self.history[opponent_player])
        if state not in self.q_table:
            self.q_table[state] = {"Cooperate": 0, "Betray": 0}

        if mode == 'train':
            prev_q_value = self.q_table[prev_state][self.last_action]
            max_q_value = max(self.q_table[state].values())
            new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value* (1-done)- prev_q_value)
            self.q_table[prev_state][self.last_action] = new_q_value
            self.save_q_table(os.path.join(self.output_path, "q_table.pkl"))


    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
