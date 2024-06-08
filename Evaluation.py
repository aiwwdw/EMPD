# Import necessary libraries
import matplotlib.pyplot as plt
import os
import pickle
from Players import Generous, Selfish, RandomPlayer, CopyCat, Grudger, Detective, Simpleton, Copykitten
from agents import *
from tqdm import tqdm
from agents.PPOagent import *
from DQN_train_v1 import save_q_table, load_q_table
from game import Game
import pickle
import matplotlib.pyplot as plt


def main():

    # episode 총 게임 수
    # max episode len 게임이 끝이 안나면 제한
    # round 한 게임 안에서 죽이는 사이클 내부 자체적으로 몇판씩 싸우나
    ############################################################################
    episode_num = 20 # number of episodes 600
    max_episode_len = 10 # maximum length of episode
    round = 20
    epsilon = 0.4 # init epsilon
    warmup_t = 20 # warmup time 200
    decay_rate = 0.98 # 100: 0.98, 1000: 0.997ßß # epsilon decay rate
    threshold = 0.1 # 0.1
    plot_num = 3 # number of plot to draw
    num_replace = 0

    # copycat selfish generous grudger detective simpleton copykitten random
    original_player_num = [1,1,1,1,1,1,1,1]
    # rlplayer smarty q_learning q_learning_business DQN LSTMDQN PPO
    rl_player_num = [0,0,0,0,1,0,0]

    history_length = 3 # 아직 연결 안됨
    test_title = '0608_dqn_1'
    plot = True # plot 할지 말지

    reward = [2,3,-1,0] # 수정 말기
    ############################################################################

    output_path = f'{test_title}_episode_{episode_num}_max_len_{max_episode_len}_epsilon_{epsilon}_replace_{num_replace}'
    output_path = os.path.join('./results', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize q_tables table based 모델의 정보 가져오기
    save_q_table(os.path.join(output_path, "simple_q_table.pkl"))
    save_q_table(os.path.join(output_path, "q_table.pkl"))
    save_q_table(os.path.join(output_path, "smarty_table.pkl"))

    # dqn 모델의 정보 저장하기
    dqn = DQN(f"DQN Player {0}", 0, output_path=output_path)
    dqn.q_network._initialize_weights()
    dqn.save_model(path=output_path)

    # data for plot
    plot_survived_round = []
    score_dict = {}
    born_dict = {}
    # episode_num 다 저장하면 너무 많아서 plot_num 만큼만 저장
    step = episode_num // plot_num
    plot_step = [ i * step for i in range(plot_num)] + [episode_num-1]

    for idx, _ in enumerate(tqdm(range(episode_num))):
        game = Game(
            output_path=output_path,
            mode='train',
            reward = reward,
            round = round, 
            replace=num_replace,
            original_player_num = original_player_num,
            rl_player_num = rl_player_num,
            epsilon = epsilon,
            history_length = history_length
            )
        game.create_players()
        

        survived_round = 1
        # Rollout the episode until max_episode_len
        for i in tqdm(range(max_episode_len)):
            if len(set(type(player) for player in game.players)) > 1:
                
                game.epsilon = epsilon
                game.start()

                ########### 게임 기록 저장 #############################
                for player in game.players:
                    if player.name not in score_dict:
                        score_dict[player.name] = [player.money]
                        born_dict[player.name] = i
                    else:
                        score_dict[player.name].append(player.money)
                #####################################################

                done = game.next_generation(dqn)
                if done:
                    survived_round = i + 1
                    break
                game.reset_player_money()

       
        ########### Save data for plot #############################
        if plot:
            plot_survived_round.append(survived_round)
            if not os.path.exists(os.path.join(output_path, "dict")):
                os.makedirs(os.path.join(output_path, "dict"))
            if idx in plot_step:
                score_dict_filename = 'score_dict_' + str(idx) + '.pkl'
                score_dict_filename = os.path.join(output_path, "dict", score_dict_filename)
                born_dict_filename = 'born_dict_' + str(idx) + '.pkl'
                born_dict_filename = os.path.join(output_path, "dict", born_dict_filename)
                
                with open(score_dict_filename, 'wb') as f:
                    pickle.dump(score_dict, f)
                with open(born_dict_filename, 'wb') as f:
                    pickle.dump(born_dict, f)
        #############################################################


        ############# print 부분 ######################################
        print(f"epsilon : {epsilon}\n")
        print("Average score")
        print(f"Survived round : {survived_round}")
        for player in score_dict.keys():
            print(f"{player}: {sum(score_dict[player])/len(score_dict[player]):.2f}")
        #############################################################
        
        score_dict = {}
        born_dic = {}

        if idx % 5 == 0:
            dqn.save_model(path=output_path)

        if idx >= warmup_t:
            epsilon = max(threshold, epsilon * decay_rate)


    # plot 활성화 여부
    if plot:   
        # Plot survived_round
        plt.figure(figsize=(10, 6))
        plt.plot(plot_survived_round)

        plt.xlabel('Episode')
        plt.ylabel('Round Number')
        plt.title('Survived Round for each episode')
        # plt.legend(loc='upper left')

        score_plot_filename = 'plot_survived_round.png'
        score_plot_filename = os.path.join(output_path, "plot", score_plot_filename)
        if not os.path.exists(os.path.join(output_path, "plot")):
            os.makedirs(os.path.join(output_path, "plot"))
        plt.savefig(score_plot_filename)

        for idx in plot_step:
            score_dict_filename = 'score_dict_' + str(idx) + '.pkl'
            score_dict_filename = os.path.join(output_path, "dict", score_dict_filename)
            with open(score_dict_filename, 'rb') as f:
                score_dict = pickle.load(f)

            born_dict_filename = 'born_dict_' + str(idx) + '.pkl'
            born_dict_filename = os.path.join(output_path, "dict", born_dict_filename)
            with open(born_dict_filename, 'rb') as f:
                born_dict = pickle.load(f)

            # Plot the scores
            plt.figure(figsize=(10, 6))
            for player in score_dict:
                # if "DQN" in player:
                x_value = list(range(born_dict[player], born_dict[player]+len(score_dict[player])))
                plt.plot(x_value, score_dict[player], label=player)

            plt.xlabel('Round')
            plt.ylabel('Score')
            plt.title('Scores of Players in Episode ' + str(idx))
            plt.legend(loc='upper left')

            score_plot_filename = 'score_plot_' + str(idx) + '.png'
            score_plot_filename = os.path.join(output_path, "plot", score_plot_filename)
            plt.savefig(score_plot_filename)

if __name__ == "__main__":
    main()