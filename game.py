import random
from Players import Generous,Selfish,RandomPlayer,CopyCat,Grudger,Detective,Simpleton,Copykitten
from RLagent import *

class Game:
    def __init__(self, num_players, num_rounds, num_replace,num_generous,num_selfish,num_copycat,num_grudger,num_detective,num_simpleton,num_copykitten,num_random,num_smart,num_smarty,ch_ch,c_c,c_ch,ch_c):
        self.players = []
        self.num_rounds = num_rounds
        self.num_players = num_players
        self.num_replace = num_replace
        self.num_generous = num_generous
        self.num_selfish = num_selfish
        self.num_copycat = num_copycat
        self.num_grudger = num_grudger
        self.num_detective = num_detective
        self.num_simpleton = num_simpleton
        self.num_copykitten = num_copykitten
        self.num_random = num_random
        self.num_smart = num_smart
        self.num_smarty = num_smarty
        self.ch_ch = ch_ch
        self.c_c = c_c
        self.c_ch =c_ch
        self.ch_c= ch_c
        self.create_players()
        self.player_names = [player.__class__.__name__ for player in self.players]

    def create_players(self):
        
        for i in range(self.num_generous):
            self.players.append(Generous(f"Generous Player {i+1}"))
        for i in range(self.num_selfish):
            self.players.append(Selfish(f"Selfish Player {i+1}"))
        for i in range(self.num_copycat):
            self.players.append(CopyCat(f"CopyCat Player {i+1}"))
        for i in range(self.num_grudger):
            self.players.append(Grudger(f"Grudger Player {i+1}"))
        for i in range(self.num_detective):
            self.players.append(Detective(f"Detective Player {i+1}"))
        for i in range(self.num_simpleton):
            self.players.append(Simpleton(f"Simpleton Player {i+1}"))
        for i in range(self.num_copykitten):
            self.players.append(Copykitten(f"Copykitten Player {i+1}"))
        for i in range(self.num_random):
            self.players.append(RandomPlayer(f"RandomPlayer Player {i+1}"))
        for i in range(self.num_smart):
            self.players.append(RLPlayer(f"RLPlayer Player {i+1}"))
        for i in range(self.num_smarty):
            self.players.append(Smarty(f"Smarty Player {i+1}"))     
        
    def start(self):
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                player1 = self.players[i]
                player2 = self.players[j]

                player1_last_action = "Cooperate"
                player2_last_action = "Cooperate"


                for round_number in range(1, self.num_rounds + 1):
                    action1 = player1.perform_action(player2_last_action, round_number)
                    action2 = player2.perform_action(player1_last_action, round_number)
                    


                    if isinstance(player1, RLPlayer):
                        reward1 = self.get_reward(action1, action2)
                        player1.update_q_table(reward1)
                    if isinstance(player2, RLPlayer):
                        reward2 = self.get_reward(action2, action1)
                        player2.update_q_table(reward2)
                    
                    if isinstance(player1, Smarty):
                        reward1 = self.get_reward(action1, action2)
                        player1.update_q_table(reward1)
                    if isinstance(player2, Smarty):
                        reward2 = self.get_reward(action2, action1)
                        player2.update_q_table(reward2)

                    player1.money += self.get_reward(action1, action2)
                    player2.money += self.get_reward(action2, action1)

                    player1_last_action = action1  
                    player2_last_action = action2

    def get_reward(self, action1, action2):
        if action1 == "Cooperate" and action2 == "Cooperate":
            return self.c_c
        elif action1 == "Cooperate" and action2 == "Betray":
            return self.c_ch
        elif action1 == "Betray" and action2 == "Cooperate":
            return self.ch_c
        elif action1 == "Betray" and action2 == "Betray":
            return self.ch_ch 
    def show_result(self):
        result = "Final Results:\n"
        for player in self.players:
            result += f"{player.name}: {player.money}\n"
        return result

    def next_generation(self):
        very_poors=[]
        reaches = []
        poors = []
        if len(self.players) > self.num_replace:
            money_values = {player.money for player in self.players}
            money_values = sorted(money_values)
            i=0
            while len(poors) + len(very_poors) <self.num_replace:
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
        
            for player in reaches[:self.num_replace]:
                random_number = random.randint(1, 50)
                if random_number == 26:
                    # player_names = [player.__class__.__name__ for player in self.players]
                    players=set(self.player_names)
                    random_player = random.sample(players, 1)[0]
                    if random_player =='CopyCat':
                        new_players.append(CopyCat(f"CopyCat Player {self.num_copycat + 1}"))
                        self.num_copycat += 1
                    elif random_player =='Selfish':
                        new_players.append(Selfish(f"Selfish Player {self.num_selfish + 1}"))
                        self.num_selfish += 1
                    elif random_player =='Generous':
                        new_players.append(Generous(f"Generous Player {self.num_generous + 1}"))
                        self.num_generous += 1
                    elif random_player =='Grudger':
                        new_players.append(Grudger(f"Grudger Player {self.num_grudger + 1}"))
                        self.num_grudger += 1
                    elif random_player =='Detective':
                        new_players.append(Detective(f"Detective Player {self.num_detective + 1}"))
                        self.num_detective += 1
                    elif random_player =='Simpleton':
                        new_players.append(Simpleton(f"Simpleton Player {self.num_simpleton + 1}"))
                        self.num_simpleton += 1  
                    elif random_player =='Copykitten':
                        new_players.append(Copykitten(f"Copykitten Player {self.num_copykitten + 1}"))
                        self.num_copykitten += 1  
                    elif random_player =='RandomPlayer':
                        new_players.append(RandomPlayer(f"RandomPlayer Player {self.num_random + 1}"))
                        self.num_random += 1
                    elif random_player =='RLPlayer':
                        new_players.append(RLPlayer(f"RLPlayer Player {self.num_smart + 1}"))
                        self.num_smart += 1 
                    elif random_player =='Smarty':
                        new_players.append(Smarty(f"Smarty Player {self.num_smarty + 1}"))
                        self.num_smarty += 1 
                else:
                    if isinstance(player, CopyCat):
                        new_players.append(CopyCat(f"CopyCat Player {self.num_copycat + 1}"))
                        self.num_copycat += 1
                    elif isinstance(player, Selfish):
                        new_players.append(Selfish(f"Selfish Player {self.num_selfish + 1}"))
                        self.num_selfish += 1
                    elif isinstance(player, Generous):
                        new_players.append(Generous(f"Generous Player {self.num_generous + 1}"))
                        self.num_generous += 1
                    elif isinstance(player, Grudger):
                        new_players.append(Grudger(f"Grudger Player {self.num_grudger + 1}"))
                        self.num_grudger += 1
                    elif isinstance(player, Detective):
                        new_players.append(Detective(f"Detective Player {self.num_detective + 1}"))
                        self.num_detective += 1
                    elif isinstance(player, Simpleton):
                        new_players.append(Simpleton(f"Simpleton Player {self.num_simpleton + 1}"))
                        self.num_simpleton += 1  
                    elif isinstance(player, Copykitten):
                        new_players.append(Copykitten(f"Copykitten Player {self.num_copykitten + 1}"))
                        self.num_copykitten += 1  
                    elif isinstance(player, RandomPlayer):
                        new_players.append(RandomPlayer(f"RandomPlayer Player {self.num_random + 1}"))
                        self.num_random += 1
                    elif isinstance(player, RLPlayer):
                        new_players.append(RLPlayer(f"RLPlayer Player {self.num_smart + 1}"))
                        self.num_smart += 1    
                    elif isinstance(player, Smarty):
                        new_players.append(Smarty(f"Smarty Player {self.num_smarty + 1}"))
                        self.num_smarty += 1

            self.players = [player for player in self.players if player not in very_poors]+new_players

    def reset_player_money(self):
        for player in self.players:
            player.money = 0        

    def announce_winner(self):
        player = self.players[0]
        if isinstance(player, CopyCat):
            return "Winners are COPYCATS"
        elif isinstance(player, Selfish):
            return "Winners are SELFISHES"
        elif isinstance(player, Generous):
            return "Winners are GENEROUSES"
        elif isinstance(player, Grudger):
            return "Winners are GRUDGERS"
        elif isinstance(player, Detective):
            return "Winners are DETECTIVES"
        elif isinstance(player, Simpleton):
            return "Winners are Simpleton"
        elif isinstance(player, Copykitten):
            return "Winners are Copykitten"
        elif isinstance(player, RandomPlayer):
            return "Winners are RandomPlayer"
        elif isinstance(player, RLPlayer):
            return "Winners are Smarts"
        elif isinstance(player, Smarty):
            return "Winners are 2nd Smarts"
