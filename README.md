## File Explanation

- **trust_evolution.py**: Running game.py using original players

- **train_and_eval_dqn.py**: Training and evaluating dqn using game.py
- **train_and_eval_q_learning.py**: Training and evaluating q-learning using game.py

- **game.py**: Environment of "The Evolution of Trust"

- /agents
    - **original_players.py**: origin agents refer from "The Evolution of Trust"
    - **rl_agent.py**: RL_agent and Smarty agent
    - **dqn_agent.py**: DQN model
    - **q_learning_agent.py**: Q-learning model



## How to Run the Code

### Q-learning
Change parameter of **train_and_eval_q_learning.py** and **q_learning_agent.py**

``` shell script
python train_and_eval_q_learning.py
```

### DQN
Change parameter of **train_and_eval_dqn.py** and **dqn_agent.py**

``` shell script
python train_and_eval_dqn.py
```

