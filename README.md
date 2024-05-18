# 정리

## 상황 정리

next_generation, reset_player_money 부분 하는중

## 내용 정리

### 기본 변수

num_rounds : 한 사이클 내부에서 같은 사람과 몇판할지

ch_Ch, c_c, c_ch, ch_c = 이름부터 거지같은데, 죄수의 딜레마 중 줄 보상의 크기 - ch가 배반을 의미 왼쪽이 내 선택

이거 변수 입력받는 부분 없애고 하드 코드함 [0,2,-1,3]

### 게임 진행시

RLPlayer에 대해서는 학습을 위해 update_q_table 실행

## 의논 사항

모든 플레이어 게임 시작시, 가장 최근 action을 Cooperate이라 가정하고 시작한다
