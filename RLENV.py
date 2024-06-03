import random
from Players import Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
from RLagent import *
class Game:
    def __init__(self):

        self.num_rounds = 10
        self.num_replace = 2
        self.ch_Ch = 0 # ch가 배반을 의미 왼쪽이 내 선택
        self.c_c = 2
        self.c_ch = -1
        self.ch_c = 3
        self.player_num = 1

        # Player Setting
        self.players = []
        self.num_players = 0
        self.num_players_left = 0
        
        # player 종류별 숫자
        self.num_copycat=0
        self.num_selfish=0
        self.num_generous=0
        self.num_grudger=0
        self.num_detective=0
        self.num_simpleton=0
        self.num_copykitten=0
        self.num_random=0
        self.num_rlplayer = 0
        self.num_smarty = 0

       
        # 게임 전적 기록 (1,2)는 player1과 player2의 게임 기록.
        self.history_dic = {}

    def create_players(self, num_players = 20, original_player_num = [2,2,2,2,2,2,2,2,2,2] ):
        while True:
            try:
                self.num_players = num_players
                self.num_players_left = self.num_players
                self.num_generous = original_player_num[0]
                self.num_selfish = original_player_num[1]
                self.num_copycat = original_player_num[2]
                self.num_grudger = original_player_num[3]
                self.num_detective = original_player_num[4]
                self.num_simpleton = original_player_num[5]
                self.num_copykitten = original_player_num[6]
                self.num_random = original_player_num[7]
                self.num_rlplayer = original_player_num[8]
                self.num_smarty = original_player_num[9]
                break
            except ValueError:
                print("Please enter a valid number.")
        num = self.player_num;
        for i in range(self.num_copycat):
            self.players.append(CopyCat(f"CopyCat Player {i+1}", num))
            num += 1
        for i in range(self.num_selfish):
            self.players.append(Selfish(f"Selfish Player {i+1}", num))
            num += 1
        for i in range(self.num_generous):
            self.players.append(Generous(f"Generous Player {i+1}", num))
            num += 1
        for i in range(self.num_grudger):
            self.players.append(Grudger(f"Grudger Player {i+1}", num))
            num += 1
        for i in range(self.num_detective):
            self.players.append(Detective(f"Detective Player {i+1}", num))
            num += 1
        for i in range(self.num_simpleton):
            self.players.append(Simpleton(f"Simpleton Player {i+1}", num))
            num += 1
        for i in range(self.num_copykitten):
            self.players.append(Copykitten(f"Copykitten Player {i+1}", num))
            num += 1
        for i in range(self.num_random):
            self.players.append(RandomPlayer(f"RandomPlayer Player {i+1}", num)) 
            num += 1   
        for i in range(self.num_rlplayer): # Add this
            self.players.append(RLPlayer(f"RLPlayer Player {i+1}", num))
            num += 1
        for i in range(self.num_smarty): # Add this
            self.players.append(Smarty(f"Smarty Player {i+1}", num))
            num += 1
        self.player_num = num

    def start(self):
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                player1 = self.players[i]
                player2 = self.players[j]
                player1_num = player1.num
                player2_num = player2.num
                player_tuple = (player2.num , player1.num) if (player1.num > player2.num) else (player1.num , player2.num)
                
                
                # 가장 최근 action의 정의
                player1_last_action = "Cooperate"
                player2_last_action = "Cooperate" 
                # 이전 사이클 내에서 전적 있을시, 최근 action 수정 
                if player_tuple in self.history_dic:
                    if player_tuple[0] == player1.num:
                        player1_last_action = self.history_dic[player_tuple][-1][0]
                        player2_last_action = self.history_dic[player_tuple][-1][1]
                    else:
                        player1_last_action = self.history_dic[player_tuple][-1][1]
                        player2_last_action = self.history_dic[player_tuple][-1][0]


                
                for round_number in range(1, self.num_rounds + 1):
                    
                    action1 = player1.perform_action(player1_last_action, player2_last_action, round_number, player2_num)
                    action2 = player2.perform_action(player2_last_action, player1_last_action, round_number, player1_num)


                    # History 저장
                    a1 = 1 if action1 == "Cooperate" else 0
                    a2 = 1 if action2 == "Cooperate" else 0
                    if player_tuple in self.history_dic:
                        self.history_dic[player_tuple].append((a1,a2))
                    else:
                        self.history_dic[player_tuple] = [(a1,a2)]


                    reward1 = self.get_reward(action1, action2)
                    reward2 = self.get_reward(action2, action1)

                    # Train
                    if isinstance(player1, RLPlayer) or isinstance(player1, Smarty):
                        player1.update_q_table(reward1)
                    if isinstance(player2, RLPlayer) or isinstance(player2, Smarty):
                        player2.update_q_table(reward2)
                
                    # Update players' money based on actions and payoffs
                    player1.money += reward1
                    player2.money += reward2

                    player1_last_action = action1
                    player2_last_action = action2

                    # print(f"{player1.name} action: {action1}, {player2.name} action: {action2}")
                    # print(f"{player1.name} earn money: {self.get_reward(action1, action2)}, {player2.name} earn money: {self.get_reward(action2, action1)}")
                    # print(f"{player1.name} final money: {player1.money}, {player2.name} final money: {player2.money}")
                    
    def get_reward(self, action1, action2):
        if action1 == "Cooperate" and action2 == "Cooperate":
            return self.c_c
        elif action1 == "Cooperate" and action2 == "Betray":
            return self.c_ch
        elif action1 == "Betray" and action2 == "Cooperate":
            return self.ch_c
        elif action1 == "Betray" and action2 == "Betray":
            return self.ch_Ch 
  
    def show_result(self):
        print("Final Results:")
        for player in self.players:
            print(f"Player Num - {player.num}, {player.name}: {player.money}")
    
    def next_generation(self):
        very_poors=[]
        reaches = []
        poors = []
        if len(self.players) > self.num_replace:
            money_values = {player.money for player in self.players}
            money_values = sorted(money_values)
            i=0
            while len(poors) + len(very_poors) < self.num_replace:
                if len(money_values)>1:
                    min_money=min(money_values)
                    money_values.remove(min_money)
                else:
                    min_money=money_values[0]
                
                for player in self.players:
                    if player.money == min_money:
                        poors.append(player)
                if len(poors)+ len(very_poors)<self.num_replace:
                    for i in range(len(poors)):
                        element=poors.pop(-1)
                        very_poors.append(element)
                    poors=[]
                else:
                    break
                i+=1
            random.shuffle(poors)
            for i in range(len(very_poors), self.num_replace):
                element = poors.pop(-1)
                very_poors.append(element)

            j=-1
            
            while len(self.players)-len(very_poors)+len(reaches)<self.num_players:
                if len(money_values)>1:
                    max_money=max(money_values)
                    money_values.remove(max_money)
                else:
                    max_money=money_values[0]
                for player in self.players:
                    if player.money == max_money:
                        reaches.append(player)
                        
                j+=1
            random.shuffle(reaches)
            new_players = []
            num = self.player_num
            for player in reaches[:self.num_replace]:
                if isinstance(player, CopyCat):
                    new_players.append(CopyCat(f"CopyCat Player {self.num_copycat + 1}", num))
                    num += 1
                    self.num_copycat += 1
                elif isinstance(player, Selfish):
                    new_players.append(Selfish(f"Selfish Player {self.num_selfish + 1}", num))
                    num += 1
                    self.num_selfish += 1
                elif isinstance(player, Generous):
                    new_players.append(Generous(f"Generous Player {self.num_generous + 1}", num))
                    num += 1
                    self.num_generous += 1
                elif isinstance(player, Grudger):
                    new_players.append(Grudger(f"Grudger Player {self.num_grudger + 1}", num))
                    num += 1
                    self.num_grudger += 1
                elif isinstance(player, Detective):
                    new_players.append(Detective(f"Detective Player {self.num_detective + 1}", num))
                    num += 1
                    self.num_detective += 1
                elif isinstance(player, Simpleton):
                    new_players.append(Simpleton(f"Simpleton Player {self.num_simpleton + 1}", num))
                    num += 1
                    self.num_simpleton += 1  
                elif isinstance(player, Copykitten):
                    new_players.append(Copykitten(f"Copykitten Player {self.num_copykitten + 1}", num))
                    num += 1
                    self.num_copykitten += 1  
                elif isinstance(player, RandomPlayer):
                    new_players.append(RandomPlayer(f"RandomPlayer Player {self.num_random + 1}", num))
                    num += 1
                    self.num_random += 1    
                elif isinstance(player, RLPlayer):
                    new_players.append(RLPlayer(f"RLPlayer Player {self.num_rlplayer + 1}", num))
                    num += 1
                    self.num_rlplayer += 1    
                elif isinstance(player, Smarty):
                    new_players.append(Smarty(f"Smarty Player {self.num_smarty + 1}", num))
                    num += 1
                    self.num_smarty += 1
            self.player_num = num;
            self.players = [player for player in self.players if player not in very_poors]+new_players
            for deleted_player in very_poors:
                deleted_player_num = deleted_player.num
                for i in range(self.player_num):
                    delete_tuple = (deleted_player_num,i) if deleted_player_num < i else (i, deleted_player_num)
                    if delete_tuple in self.history_dic:
                        del self.history_dic[delete_tuple]

    def reset_player_money(self):
        for player in self.players:
            player.money = 0        
    
    def announce_winner(self):
        player=self.players[0]
        if isinstance(player, CopyCat):
            print("Winners are COPYCATS")     
        elif isinstance(player, Selfish):
            print("Winners are SELFISHES")
        elif isinstance(player, Generous):
            print("Winners are GENEROUSES")
        elif isinstance(player, Grudger):
            print("Winners are GRUDGERS")
        elif isinstance(player, Detective):
            print("Winners are DETECTIVES")
        elif isinstance(player, Copykitten):
            print("Winners are Copykitten")
        elif isinstance(player, Simpleton):
            print("Winners are Simpleton")
        elif isinstance(player, RandomPlayer):
            print("Winners are RandomPlayer")
        elif isinstance(player, RLPlayer):
            print("Winners are Smarts")
        elif isinstance(player, Smarty):
            print("Winners are 2nd Smarts")


# GUI는 return 하는 반면 여기는 다 print로 처리
#
def main():
    game = Game()
    game.create_players()
    
    c=1
    while len(set(type(player) for player in game.players)) > 1 and c<100:
        
        print(f"round number {c} started")
        
        game.start()
        
        game.show_result()

        game.next_generation()
        game.reset_player_money()
        c+=1

    # game.announce_winner()


if __name__ == "__main__":
    main()
