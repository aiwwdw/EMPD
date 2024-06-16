import random
from agents.orignal_players import Player,Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
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
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cuda') 

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]


class ActorCritic(nn.Module):
    def __init__(self, input_dim, history_length):
        super(ActorCritic, self).__init__()

        self.history_num = history_length
        
        self.fc1_actor = nn.Linear(input_dim, 64)
        self.conv1_actor = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 10), stride=2, padding=1)
        self.conv2_actor = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 10), stride=2, padding=1)
        self.conv3_actor = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 10), stride=2, padding=1)
        self.fc2_actor = nn.Linear(512, 2)


        self.fc1_critic = nn.Linear(input_dim, 64)
        self.conv1_critic = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 10), stride=2, padding=1)
        self.conv2_critic = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 10), stride=2, padding=1)
        self.conv3_critic = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 10), stride=2, padding=1)
        self.fc2_critic = nn.Linear(512, 1)
        
    def critic(self, state):
        batch_size = state.size(0)
        state = self.fc1_critic(state)
        state = state.unsqueeze(0)
        state = F.relu(self.conv1_critic(state))
        state = F.relu(self.conv2_critic(state))
        state = F.relu(self.conv3_critic(state))
        state = F.adaptive_avg_pool2d(state, (1, 1))
        state = state.view(batch_size, -1)
        value = self.fc2_critic(state)
        return value
        
    def actor(self, state):
        batch_size = state.size(0)
        state = self.fc1_actor(state)

        state = state.unsqueeze(0)

        state = F.relu(self.conv1_actor(state))
        state = F.relu(self.conv2_actor(state))
        state = F.relu(self.conv3_actor(state))
        state = F.adaptive_avg_pool2d(state, (1, 1))

        state = state.view(batch_size, -1)
        state = self.fc2_actor(state)
        action_probs = F.softmax(state, dim=-1)

        return action_probs
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


    def forward(self):
        raise NotImplementedError


class PPO(Player):
    def __init__(self, name, num, K_epochs=3, eps_clip = 0.1, lr_actor=0.001, lr_critic=0.001, gamma=0.99, eplsilon= 0.1, history_length=10, input_dim=5, history_dim=10, output_path='./results'):
        super().__init__(name, num)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.history_length = history_length 
        self.gamma = gamma  # Discount factor
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(input_dim, history_length).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        # self.optimizer = torch.optim.Adam([
        #                 {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #                 {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        #             ])

        self.policy_old = ActorCritic(input_dim, history_length).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.history = {}
        self.prev_history = {}


    def perform_action(self, agent_last_action, opponent_last_action, round_number, opponent_player, epsilon):
        base_list = [1, 0, 0, 0, 0]

        if opponent_player not in self.history:
            self.history[opponent_player] = [base_list] * self.history_length
            self.prev_history[opponent_player] = [base_list] * self.history_length

        self.prev_history[opponent_player] = copy.deepcopy(self.history[opponent_player])
        if (agent_last_action, opponent_last_action) == ("Cooperate", "Cooperate"):
            current_state = [0, 0, 0, 0, 1]
        elif (agent_last_action, opponent_last_action) == ("Cooperate", "Betray"):
            current_state = [0, 0, 0, 1, 0]
        elif (agent_last_action, opponent_last_action) == ("Betray", "Cooperate"):
            current_state = [0, 0, 1, 0, 0]
        elif (agent_last_action, opponent_last_action) == ("Betray", "Betray"):
            current_state = [0, 1, 0, 0, 0]
        else:
            print("Wrong last action tuples")
            sys.exit(1)
        
        self.history[opponent_player].append(current_state)
        self.history[opponent_player].pop(0)
        
        state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
        
        action = "Cooperate" if action == 1 else "Betray"
        return action

    def update_q_table(self, action, reward, oppenent_playe, player1_last_action, player2_last_action):
        # Monte Carlo estimate of returns

        if action == "Cooperate":
            action = 1
        elif action == "Betray":
            action = 0
        
        rewards = []
        discounted_reward = 0
        for reward in zip(reversed(self.buffer.rewards)):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).squeeze(0).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        print(old_states.size())
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    



#-------------------------------------------------------
# class PPO(Player):
#     def __init__(self, name, num, alpha=0.001, gamma=0.99, epsilon=0.1, history_length=10, input_dim=5, history_dim=10, output_path='./results'):
#         super().__init__(name, num)
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.history_length = history_length
#         self.memory = deque(maxlen=1000)
#         self.batch_size = 128
#         self.update_every = 40
#         self.step = 0
#         self.history = {}
#         self.prev_history = {}

#         # self.policy_network = PolicyNetwork(input_dim, history_dim).to('cuda')
#         # self.value_network = ValueNetwork(input_dim, history_dim).to('cuda')
#         self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.alpha)
#         self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.alpha)
#         self.loss_fn = nn.MSELoss()
        
#     def perform_action(self, agent_last_action, opponent_last_action, round_number, opponent_player, epsilon):
#         base_list = [1, 0, 0, 0, 0]

#         if opponent_player not in self.history:
#             self.history[opponent_player] = [base_list] * self.history_length
#             self.prev_history[opponent_player] = [base_list] * self.history_length

#         self.prev_history[opponent_player] = copy.deepcopy(self.history[opponent_player])
#         if (agent_last_action, opponent_last_action) == ("Cooperate", "Cooperate"):
#             current_state = [0, 0, 0, 0, 1]
#         elif (agent_last_action, opponent_last_action) == ("Cooperate", "Betray"):
#             current_state = [0, 0, 0, 1, 0]
#         elif (agent_last_action, opponent_last_action) == ("Betray", "Cooperate"):
#             current_state = [0, 0, 1, 0, 0]
#         elif (agent_last_action, opponent_last_action) == ("Betray", "Betray"):
#             current_state = [0, 1, 0, 0, 0]
#         else:
#             print("Wrong last action tuples")
#             sys.exit(1)
        
#         self.history[opponent_player].append(current_state)
#         self.history[opponent_player].pop(0)
        
#         state = torch.FloatTensor(self.history[opponent_player]).unsqueeze(0).to('cuda')
#         with torch.no_grad():
#             action_probs = self.policy_network(state)
#         action = "Cooperate" if torch.multinomial(action_probs, 1).item() == 0 else "Betray"
#         return action
    
#     def update(self, action, reward, opponent_player):
#         if action == "Cooperate":
#             action = 0
#         elif action == "Betray":
#             action = 1
#         self.memory.append((self.prev_history[opponent_player], action, reward, self.history[opponent_player]))  # remember
        
#         if len(self.memory) < self.batch_size:
#             return

#         batch = random.sample(self.memory, self.batch_size)
#         state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

#         state_batch = torch.FloatTensor(state_batch).to('cuda')
#         action_batch = torch.LongTensor(action_batch).to('cuda')
#         reward_batch = torch.FloatTensor(reward_batch).to('cuda')
#         next_state_batch = torch.FloatTensor(next_state_batch).to('cuda')

#         # Calculate advantages
#         with torch.no_grad():
#             current_values = self.value_network(state_batch)
#             next_values = self.value_network(next_state_batch)
#             advantages = reward_batch + (self.gamma * next_values) - current_values
        
#         # Calculate policy loss
#         old_action_probs = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
#         new_action_probs = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
#         ratios = new_action_probs / old_action_probs
#         surr1 = ratios * advantages
#         surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
#         policy_loss = -torch.min(surr1, surr2).mean()
        
#         # Calculate value loss
#         current_values = self.value_network(state_batch).squeeze(1)
#         value_loss = self.loss_fn(current_values, reward_batch)
        
#         # Update policy network
#         self.policy_optimizer.zero_grad()
#         policy_loss.backward()
#         self.policy_optimizer.step()

#         # Update value network
#         self.value_optimizer.zero_grad()
#         value_loss.backward()
#         self.value_optimizer.step()
        
#         self.step += 1