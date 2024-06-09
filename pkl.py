import pandas as pd
import pickle
from agents import *

def print_pkl():
    # .pkl 파일 경로 설정
    file_path = '/root/EMPD/results/q_learning_small_episode_10000_max_len_1_epsilon_1_replace_0/q_table.pkl'

    # .pkl 파일 열기
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 데이터가 DataFrame인지 확인하고 출력
    print(data)



def print_dqn_result():

    his_len = 5
    checkpoints = '/root/EMPD/results/q_learning_small_episode_10000_max_len_1_epsilon_1_replace_0/q_table.pkl'


    history = {}
    no = [1,0,0]
    cc = [0,1,1]
    cb = [0,1,0]
    bc = [0,0,1]
    bb = [0,0,0]
    

    back = [cc,cc]


    his_len -= len(back)
    history[1] = [no] * his_len  + back
    dqn = DQN(f"DQN {0}", 0)
    dqn.load_model(path = checkpoints)
    b, c = dqn.check_network(1, history, epsilon=0)
    print("Betray: ", b.item())
    print("Cooperate: ",c.item())


if __name__ == "__main__":

    # print_dqn_result()
    print_pkl()


{(): {'Cooperate': 7.856299999999987, 'Betray': 8.856299999999983}, 
(('Cooperate', 'Cooperate'),): {'Cooperate': 5.506999999999989, 'Betray': 6.506999999999989}, 
(('Cooperate', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 4.699999999999994, 'Betray': 7.2299999999999915},
(('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate')): {'Cooperate': 4.699999999999994, 'Betray': 2.9999999999999982},
(('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0},
(('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, 
(('Cooperate', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 4.699999999999994, 'Betray': 7.2299999999999915}, 
(('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, 
(('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, 
(('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, 
(('Betray', 'Cooperate'),): {'Cooperate': 5.506999999999989, 'Betray': 6.506999999999989}, 
(('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 4.699999999999994, 'Betray': 7.2299999999999915}, 
(('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982},
(('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0},
(('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0},
(('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 4.699999999999994, 'Betray': 7.2299999999999915}, 
(('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, 
(('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0},
 (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, 
 (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, 
 (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate')): {'Cooperate': 4.699999999999994, 'Betray': 2.9999999999999982}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate')): {'Cooperate': 4.699999999999994, 'Betray': 2.9999999999999982}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate')): {'Cooperate': 4.699999999999994, 'Betray': 2.9999999999999982}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 1.9999999999999991, 'Betray': 2.9999999999999982}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': -0.9999999999999996, 'Betray': 0.0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Cooperate', 'Betray'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Cooperate', 'Cooperate'), ('Cooperate', 'Betray'), ('Betray', 'Cooperate'), ('Betray', 'Cooperate'), ('Betray', 'Betray')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Betray', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Betray', 'Cooperate')): {'Cooperate': 0, 'Betray': 0}, (('Betray', 'Cooperate'), ('Betray', 'Betray'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Cooperate'), ('Cooperate', 'Betray')): {'Cooperate': 0, 'Betray': 0}}