import random
from Players import Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
from agents import *
from tqdm import tqdm
from agents.PPOagent import *
from game import Game
# from math import cos, pi

def save_q_table(filename):
    with open(filename, 'wb') as f:
        pickle.dump({}, f)
def load_q_table(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
# GUI는 return 하는 반면 여기는 다 print로 처리
#
def main():
    
    # mode = 'eval' # train at train
    # output_path에 학습 결과 저장
    output_path = 'test_dqn_epoch_100_num_round_20_epsilon_0_4_num_replace_1'

    root_path = './results'
    output_path = os.path.join(root_path, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize q_tables
    save_q_table(os.path.join(output_path, "simple_q_table.pkl"))
    save_q_table(os.path.join(output_path, "q_table.pkl"))
    save_q_table(os.path.join(output_path, "smarty_table.pkl"))
    # dqn= 0
    dqn = DQN(f"DQN Player {0}", 0, output_path=output_path)
    dqn.q_network._initialize_weights()
    dqn.save_model(path=output_path)

    # From pretrained
    # pretrained = None
    # if pretrained != None:
    # dqn.load_model(path = os.path.join(output_path, "checkpoints.pt"))
    
    
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
                
                # print()
                # print(f"round number {i+1} started")
                # print(f"epsilon : {epsilon}")
                
                game.epsilon = epsilon
                game.start()

                
                # game.show_result()

                # TODO: mean reward 구하기, mean round도
                for player in game.players:
                    if player.name not in score_dict:
                        # score_dict[player.name] = []
                        score_dict[player.name] = [player.money]
                    else:
                        score_dict[player.name].append(player.money)

                done = game.next_generation(dqn)
                if done:
                    survived_round = i+1
                    break

                game.reset_player_money()

        print()
        print("Average score")
        # Only for replace option ON
        print(f"Survived round : {survived_round}")
        for player in score_dict.keys():
            # print(player, len(score_dict[player]))
            print(f"{player}: {sum(score_dict[player])/len(score_dict[player]):.2f}")
        score_dict = {}

        if idx % 5 == 0:
            dqn.save_model(path = output_path)

        if idx >= warmup_t:
            epsilon = max(threshold, epsilon * decay_rate)
        
        # visualize q_table
        # if idx % 20 == 0:
        #     print(f"Q_learning table: {load_q_table(os.path.join(output_path,'q_table.pkl'))}")
        
        # for player in game.players:
        #     if isinstance(player, Q_learning):
        #         print(f"Q_learning {player.num}: {player.q_table}")

    # Validation
    valid_epoch = 5
    score_dict = {}
    for idx, _ in enumerate(tqdm(range(valid_epoch))):
        game = Game(mode='test', output_path=output_path)
        game.create_players(dqn)

        # Rollout the episode until max_episode_len
        for i in range(max_episode_len):
            if len(set(type(player) for player in game.players)) > 1:
                
                print()
                print(f"round number {i+1} started")
                print(f"epsilon : {game.epsilon}")
                
                game.start()

                game.epsilon = 0
                
                game.show_result()
                # TODO: mean reward 구하기, mean round도
                for player in game.players:
                    if player.name not in score_dict:
                        score_dict[player.name] = [player.money]
                    else:
                        score_dict[player.name].append(player.money)
                    # print(f"Player Num - {player.num}, {player.name}: {player.money}")

                game.next_generation(dqn)
                game.reset_player_money()
    
    print("Average score")
    for player in score_dict.keys():
        print(f"{player}: {sum(score_dict[player])/len(score_dict[player]):.2f}")
    # game.announce_winner()


if __name__ == "__main__":
    main()
