# Evolution of Trust with Reinforcement Learning

This project explores **"The Evolution of Trust"**, a multiplayer extension of the Prisoner’s Dilemma, using **Reinforcement Learning (RL)** techniques. The main objective is to train RL agents (Q-learning and DQN models) to identify optimal cooperative strategies and adapt to dynamic multiplayer environments.

---

## Features

1. **Game Environment**: Simulates the "Evolution of Trust" as a series of multiplayer Iterated Prisoner’s Dilemma games.
2. **Reinforcement Learning Models**:
   - Q-learning: Basic RL model for strategy optimization.
   - Deep Q-Network (DQN): Advanced RL model using neural networks for state-action learning.
3. **Original Players**: Includes baseline strategies like Copycat, Generous, and Random to test against RL agents.

---

## File Structure and Explanation

### Main Files
- **`trust_evolution.py`**: Runs the game environment using original players only (baseline test).
- **`train_and_eval_dqn.py`**: Handles training and evaluation of the DQN model within the game environment.
- **`train_and_eval_q_learning.py`**: Manages training and evaluation of the Q-learning model.

### Game Environment
- **`game.py`**: Defines the rules, scoring, and player interactions for "The Evolution of Trust."

### Agent Directory (`/agents`)
- **`original_players.py`**: Implements baseline strategies such as Copycat, Generous, and Random.
- **`rl_agent.py`**: Contains RL agents, including:
  - RL_agent: A basic reinforcement learning agent.
  - Smarty: A variant of RL_agent with extended memory.
- **`dqn_agent.py`**: Implements the DQN model for the game environment.
- **`q_learning_agent.py`**: Implements the Q-learning model for the game environment.

---

## How to Run the Code

### Q-learning
1. Modify the parameters in `train_and_eval_q_learning.py` and `q_learning_agent.py` as needed.
2. Run the following command to start training and evaluation:
   ```bash
   python train_and_eval_q_learning.py

## Resources

- **[Research Paper](https://github.com/aiwwdw/EMPD/blob/main/Evolution_of_trust.pdf)**: A detailed explanation of the research, methodology, and results.
- **[Presentation Slides](https://github.com/aiwwdw/EMPD/blob/main/강화학습%20최종%20발표.pdf)**: A concise overview of the project presented in slide format.
