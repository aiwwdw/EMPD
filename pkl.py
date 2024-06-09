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
    checkpoints = '/root/EMPD/results/0_dqn_solo_episode_100_max_len_2_epsilon_1_replace_0/checkpoints.pt'


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

    print_dqn_result()
    # print_pkl()

