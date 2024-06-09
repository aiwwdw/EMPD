# 정리

## 상황 정리

next_generation, reset_player_money 부분 하는중

## 내용 정리

### 환경

TrustEvolution : RL을 제외한 player 환경

RLENV : print 기반 환경 - 이걸 메인으로

Players : RL 제외한 player 클래스

RLagent

### 기본 변수

num_rounds : 한 사이클 내부에서 같은 사람과 몇판할지

ch_Ch, c_c, c_ch, ch_c = 이름부터 거지같은데, 죄수의 딜레마 중 줄 보상의 크기 - ch가 배반을 의미 왼쪽이 내 선택

이거 변수 입력받는 부분 없애고 하드 코드함 [0,2,-1,3]

#### action 변수
- Cooperate: 1
- Betary : 0 

#### state 변수

Cooperate : 1
Betray : 0

- [1,0,0,0,0] : Nothing
- [0,1,0,0,0] : B vs B
- [0,0,1,0,0] : B vs C
- [0,0,0,1,0] : C vs B
- [0,0,0,0,1] : C vs C

### 게임 진행시

RLPlayer에 대해서는 학습을 위해 update_q_table 실행

## 의논 사항 및 할일들

현재 세팅: 

모든 플레이어 게임 시작시, 가장 최근 action을 Cooperate이라 가정하고 시작한다

한 사이클 시작시 돈을 0으로 리셋함, 이를 변경할지 말지

opservation 영역을: player들의 승패 유무로 할지, player들의 배신 협력 여부로 할지
- 둘다 배열로 만들예정인데 이걸 RL한테 어떤 데이터로 줄지 얘기
  
    - n명이  nC2(1회반복) X 반복횟수 에 대해서 
      - (player1 번호, player2 번호, player1 협력여부, player2 협력 여부)
    - n명이  nC2(1회반복) 에 대해서 
      - (player1 번호, player2 번호, 반복후 reward 값)
  

input 터미널에 입력 - 하드 코드로 변경 예정


## 실험 진행

### 1. Q-Learning

  - 1:1게임에서 detective를 제외한 모든 player을 이길 전략을 찾아냄
    - detective에서 지는 이유는 detective는 앞 4개의 상황을 보고 진행함
    - History = 3, size of q table = 85 
  
  - History = 5로 사이즈를 늘려서 실험
    - size of q table = 440
    - detective를 이겨낼 전략도 찾아냄.
    - 하지만 사이즈가 약 5정도 늘어나고, 실질적인 모든 q-table을 다 search를 구하지 못함. $5^5 \sim 3000$ 
  - 새로운 state를 제안
    - 앞 3개와 뒤 3개를 보고 전략을 짜도록 만들자
    - $$진행해야함$$
  - Q-learning에 단점으로 state space를 늘릴 수 없다는 점은 detective를 이길 전략을 없게 만듬

  - 1:n 실험에서 q-learning은 완전히 패함
    - 1:1로 싸울떄는 각 player에 대한 q_table을 만드는 것이 쉽지만,
    - 상대 player를 모르는 상황에서는 q_table이 stable이 안되서 general한 전략을 짜는데 실패.


### 2. DQN



