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
    
    # output_path에 학습 결과 저장
    output_path = 'dqn_debug'

    root_path = './results'
    output_path = os.path.join(root_path, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize q_tables
    save_q_table(os.path.join(output_path, "simple_q_table.pkl"))
    save_q_table(os.path.join(output_path, "q_table.pkl"))
    save_q_table(os.path.join(output_path, "smarty_table.pkl"))
    # dqn= 0
    dqn = nstep_DQN(f"DQN Player {0}", 0, output_path=output_path)
    dqn.q_network._initialize_weights()
    dqn.save_model(path=output_path)
    
    # number of episodes
    episode_num = 600

    # maximum length of episode
    max_episode_len = 10

    # init epsilon
    epsilon = 0.0

    # warmup time
    warmup_t = 200

    # epsilon decay rate
    decay_rate = 0.997 # 100: 0.98, 1000: 0.997ßß

    # threshold
    threshold = 0.1

    for idx, _ in enumerate(tqdm(range(episode_num))):
        # Reset the game
        game = Game(output_path=output_path)
        game.create_players()

        # Rollout the episode until max_episode_len
        for i in tqdm(range(max_episode_len)):
            if len(set(type(player) for player in game.players)) > 1:
                
                print()
                print(f"round number {i+1} started")
                print(f"epsilon : {epsilon}")
                
                game.epsilon = epsilon
                game.start()

                
                game.show_result()

                done = game.next_generation(dqn)
                if done:
                    break

                game.reset_player_money()

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

            game.next_generation(dqn)
            game.reset_player_money()

    game.announce_winner()


if __name__ == "__main__":
    main()
