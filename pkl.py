import pandas as pd
import pickle

# .pkl 파일 경로 설정
file_path = '/root/EMPD/results/pratice/q_table.pkl'

# .pkl 파일 열기
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 데이터가 DataFrame인지 확인하고 출력
print(len(data))
