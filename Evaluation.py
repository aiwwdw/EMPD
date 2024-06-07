# Import necessary libraries
import matplotlib.pyplot as plt
import os
import pickle
from Players import Generous, Selfish, RandomPlayer, CopyCat, Grudger, Detective, Simpleton, Copykitten
from agents import *
from tqdm import tqdm
from agents.PPOagent import *
from DQN_train_v1 import Game, save_q_table, load_q_table

def main():
    # mode = 'eval' # train at train
    # output_path에 학습 결과 저장
    output_path = 'dqn_epoch_100_num_round_20_epsilon_0_4_num_replace_1'

    root_path = './results'
    output_path = os.path.join(root_path, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize q_tables
    save_q_table(os.path.join(output_path, "simple_q_table.pkl"))
    save_q_table(os.path.join(output_path, "q_table.pkl"))
    save_q_table(os.path.join(output_path, "smarty_table.pkl"))

    dqn = DQN(f"DQN Player {0}", 0, output_path=output_path)
    dqn.q_network._initialize_weights()
    dqn.save_model(path=output_path)

    # number of episodes
    episode_num = 100 # 600

    # maximum length of episode
    max_episode_len = 10

    # init epsilon
    epsilon = 0.4

    # warmup time
    warmup_t = 20 #200

    # epsilon decay rate
    decay_rate = 0.98 # 100: 0.98, 1000: 0.997ßß

    # threshold
    threshold = 0.1 # 0.1

    score_dict = {}

    for idx, _ in enumerate(tqdm(range(episode_num))):
        # Reset the game
        game = Game(output_path=output_path)
        game.create_players(dqn)
        print(f"epsilon : {epsilon}")

        survived_round = 1
        # Rollout the episode until max_episode_len
        for i in tqdm(range(max_episode_len)):
            if len(set(type(player) for player in game.players)) > 1:
                
                game.epsilon = epsilon
                game.start()
                
                for player in game.players:
                    if player.name not in score_dict:
                        score_dict[player.name] = [player.money]
                    else:
                        score_dict[player.name].append(player.money)

                done = game.next_generation(dqn)
                if done:
                    survived_round = i + 1
                    break

                game.reset_player_money()

        print()
        print("Average score")
        # Only for replace option ON
        print(f"Survived round : {survived_round}")
        for player in score_dict.keys():
            print(f"{player}: {sum(score_dict[player])/len(score_dict[player]):.2f}")
        score_dict = {}

        if idx % 5 == 0:
            dqn.save_model(path=output_path)

        if idx >= warmup_t:
            epsilon = max(threshold, epsilon * decay_rate)

    # Save scores for plotting
    with open('score_dict.pkl', 'wb') as f:
        pickle.dump(score_dict, f)

if __name__ == "__main__":
    main()

# After running the above code, the next part is to create a notebook for plotting
# Below is the code for the notebook to plot the results

import pickle
import matplotlib.pyplot as plt

# Load the score dictionary
with open('score_dict.pkl', 'rb') as f:
    score_dict = pickle.load(f)

# Plot the scores
plt.figure(figsize=(10, 6))
for player in score_dict:
    plt.plot(score_dict[player], label=player)

plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Scores of Players Over Episodes')
plt.legend(loc='upper left')
plt.show()