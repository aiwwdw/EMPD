import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Players import Player,Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
from RLENV import *
import pickle

import gc


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden
    
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        output = self.linear(lstm_out)
        
        return output, self.hidden
    
class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, input_size)

        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:,t,:] = out

        return outputs

    def predict(self, inputs, target_len):
        inputs = inputs.unsqueeze(0)
        self.eval()
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out
        return outputs.detach().numpy()[0,:,0]

class windowDataset(Dataset):
    def __init__(self, y, input_window=10, output_window=1, stride=2):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

# 홀수 인자들이 학습할 수 있는 요소들, 이것들을 학습.
def main():
    
    # 데이터 정보
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traindata_given = 10 # 짝수개에 stride 2인게 홀수 번째 예측을 유도
    traindata_predicting = 1

    # data
    train = np.random.choice([0, 1], size=800) # 모델 선택, 상대 선택이 번갈아나오는 np 배열이면 성립
    test = np.random.choice([0, 1], size=traindata_given).reshape(-1, 1)
    train_dataset = windowDataset(train, input_window=traindata_given, output_window=traindata_predicting, stride=2)
    train_loader = DataLoader(train_dataset, batch_size=64)

    # 모델 설정
    model = lstm_encoder_decoder(input_size=1, hidden_size=16).to(device)
    learning_rate=0.01
    epoch = 3
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    with tqdm(range(epoch)) as tr:
        for i in tr:
            total_loss = 0.0
            for x,y in train_loader:
                optimizer.zero_grad()
                x = x.to(device).float()
                y = y.to(device).float()
                output = model(x, y, traindata_predicting, 0.6).to(device)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
            tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))

    predict = model.predict(torch.tensor(test).to(device).float(), target_len=traindata_predicting)
    print(predict)

if __name__ == "__main__":
    main()

class LSTM(Player):
    def __init__(self, name, num, learning_rate=0.01, epoch = 3, traindata_given = 10, traindata_predicting = 1):
        super().__init__(name, num)
        self.last_action = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traindata_given = traindata_given
        self.traindata_predicting = traindata_predicting
        self.models = {}
        self.learning_rate=learning_rate
        self.epoch = epoch
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {}


    def perform_action(self, recent_actions, other_player_num):
        if self.num > other_player_num:
            recent_actions = [item for sublist in recent_actions for item in (sublist[1] == "Cooperate", sublist[0] == "Cooperate")]
        else:
            recent_actions = [item for sublist in recent_actions for item in (sublist[0] == "Cooperate", sublist[1] == "Cooperate")]
        
        prefix_list = [1/2] * self.traindata_given*2 - len(recent_actions)
        recent_actions = prefix_list + recent_actions

        self.train(self,recent_actions,other_player_num)

        if other_player_num not in self.models:
            self.models[other_player_num] = lstm_encoder_decoder(input_size=1, hidden_size=16).to(self.device)
        model = self.models[other_player_num]
        action = model.predict(torch.tensor(recent_actions).to(self.device).float(), target_len=self.traindata_predicting)
        return action


    def train(self, recent_actions, other_player_num):
        
        train = recent_actions
        train_dataset = windowDataset(train, input_window=self.traindata_given, output_window=self.traindata_predicting, stride=2)
        train_loader = DataLoader(train_dataset, batch_size=64)

        model = self.models[other_player_num]
        model.train()
        for i in range(self.epoch):
            total_loss = 0.0
            for x,y in train_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                output = model(x, y, self.traindata_predicting, 0.6).to(self.device)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().item()

    def reset(self):
        for i in range(len(self.models)):
            del self.models[i]  # 모델 객체 삭제
        gc.collect()  # 가비지 컬렉터 실행
        torch.cuda.empty_cache()  # CUDA 캐시 비우기 (GPU 사용 시)
        self.opponent_history = []
        self.last_action = None
        