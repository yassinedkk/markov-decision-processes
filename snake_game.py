
import matplotlib.pyplot as plt
import numpy as np
import random as rd


def markovDecision(layout,circle):  
    """
        layout[i] = 0 if it is an ordinary square
              = 1 if it is a “restart” trap (go back to square 1)
              = 2 if it is a “penalty” trap (go back 3 steps)
              = 3 if it is a “prison” trap (skip next turn)
              = 4 if it is a “bonus” trap (play again, without waiting next turn)
            
            circle = true, if the player must land exactly on the final, goal, square 15 to win
    """
    Expec = np.zeros(15) # expected cost for each state of the game
    Dice = np.ones(14) # choice of the dice to use for each state 
    transition_prob = construct_P(circle)  
    states = range(14)
    actions = ["security", "normal", "risky"]

    #####################################################
    # begin Value Iteration algorithm #
    #####################################################
    step = 0
    while True:
        delta = 0
        for k in reversed(states):# reversed order faster, since we use backward propagation
            Values= []
            for a in actions:
                tmp = V_k(layout, transition_prob, k, a, Expec)
                Values.append(tmp)      
            best_value = min(Values)
            delta = max(delta, abs(Expec[k] - best_value))
            Expec[k] = best_value
        step+=1
        if delta < 1e-6:# stationarity check
            print("total number of step : " + str(step))
            break
    #####################################################
    # end Value Iteration algorithm #
    #####################################################
    # extract the optimal policy as Dice
    for k in states[:14]:
        dico = {}
        dico[1] = V_k(layout, transition_prob, k, "security", Expec)
        dico[2] = V_k(layout, transition_prob, k, "normal", Expec)
        dico[3] = V_k(layout, transition_prob, k, "risky", Expec)
        best_action = min(dico, key = dico.get)
        Dice[k] = best_action

    
    return [Expec[:14],Dice]

def V_k(layout, transition_prob, k, a, Expec):
    """
        Compute the expected cost for a given state k and a given action a,
        for one iteration of the value iteration algorithm.
        
        Parameters
        ----------
        layout : numpy.ndarray containing the layout of the game.
        transition_prob : dict() containing transition probability.
        k : int, current state.
        a : int, chosen action.
        Expec : numpy.ndarray, current expected cost for all state
        """
    tmp = 1# +1 for cost of doing action  
    # other state have transition_prob = 0
    # => no need to iterate over all k_prime
    for k_prime, proba in transition_prob[k][a].items():
        if a != "security" and layout[k_prime]==1:
            # it is a “restart” trap (go back to square 1)
            # => 50% to activate trap + 50% to not activate trap
            if a == "normal":
                tmp += proba/2 * Expec[k_prime] + proba/2 * Expec[0]
            else:
                tmp += proba * Expec[0]
        elif a != "security" and layout[k_prime]==2:
            # it is a “penalty” trap (go back 3 steps)
            if a == "normal":
                tmp += proba/2 * Expec[k_prime] + proba/2 * Expec[max(k_prime-3,0)]
            else:
                tmp += proba * Expec[max(k_prime-3,0)]
        elif a != "security" and layout[k_prime]==3:
            # it is a “prison” trap (skip next turn)
            # => next turn has +1 expected cost
            if a == "normal":
                tmp += proba/2 * Expec[k_prime] + proba/2 * (1 + Expec[k_prime])
            else:
                tmp += proba * (1 + Expec[k_prime])
        elif a != "security" and layout[k_prime]==4:
            # it is a “bonus” trap (play again, without waiting next turn)
            # => next turn has -1 expected cost
            if a == "normal":
                tmp += proba/2 * Expec[k_prime] + proba/2 * (Expec[k_prime] - 1)
            else:
                tmp += proba * (Expec[k_prime] - 1)
        else:
            # it is an ordinary square
            tmp += proba * Expec[k_prime]
    return tmp


# return transition proba P(k'|k,a) as P[k][a] = P(k'|k,a)
def construct_P(circle):
    # default transition without trap/bonus and without circle
    transition_prob = {
        0: {"security": {0: 0.5, 1: 0.5}, "normal": {0: 1/3.0, 1: 1/3.0, 2:1/3.0}, "risky": {0: 1/4.0, 1: 1/4.0, 2:1/4.0, 3: 1/4.0}},
        1: {"security": {1: 0.5, 2: 0.5}, "normal": {1: 1/3.0, 2: 1/3.0, 3:1/3.0}, "risky": {1: 1/4.0, 2: 1/4.0, 3:1/4.0, 4: 1/4.0}},
        2: {"security": {2: 0.5, 3: 0.25, 10: 0.25}, "normal": {2: 1/3.0, 3: 1/6.0, 10: 1/6.0, 4:1/6.0, 11:1/6.0}, "risky": {2: 1/4.0, 3: 1/8.0, 10:1/8.0, 4:1/8.0, 11:1/8.0, 5: 1/8.0, 12: 1/8.0}},
        3: {"security": {3: 0.5, 4: 0.5}, "normal": {3: 1/3.0, 4: 1/3.0, 5:1/3.0}, "risky": {3: 1/4.0, 4: 1/4.0, 5:1/4.0, 6: 1/4.0}},
        4: {"security": {4: 0.5, 5: 0.5}, "normal": {4: 1/3.0, 5: 1/3.0, 6:1/3.0}, "risky": {4: 1/4.0, 5: 1/4.0, 6:1/4.0, 7: 1/4.0}},
        5: {"security": {5: 0.5, 6: 0.5}, "normal": {5: 1/3.0, 6: 1/3.0, 7:1/3.0}, "risky": {5: 1/4.0, 6: 1/4.0, 7:1/4.0, 8: 1/4.0}},
        6: {"security": {6: 0.5, 7: 0.5}, "normal": {6: 1/3.0, 7: 1/3.0, 8:1/3.0}, "risky": {6: 1/4.0, 7: 1/4.0, 8:1/4.0, 9: 1/4.0}},
        7: {"security": {7: 0.5, 8: 0.5}, "normal": {7: 1/3.0, 8: 1/3.0, 9:1/3.0}, "risky": {7: 1/4.0, 8: 1/4.0, 9:1/4.0, 14: 1/4.0}},
        8: {"security": {8: 0.5, 9: 0.5}, "normal": {8: 1/3.0, 9: 1/3.0, 14:1/3.0}, "risky": {8: 1/4.0, 9: 1/4.0, 14:1/2.0}}, 
        9: {"security": {9: 0.5, 14: 0.5}, "normal": {9: 1/3.0, 14: 2/3.0}, "risky": {9: 1/4.0, 14: 3/4.0}},
        10: {"security": {10: 0.5, 11: 0.5}, "normal": {10: 1/3.0, 11: 1/3.0, 12:1/3.0}, "risky": {10: 1/4.0, 11: 1/4.0, 12:1/4.0, 13: 1/4.0}},
        11: {"security": {11: 0.5, 12: 0.5}, "normal": {11: 1/3.0, 12: 1/3.0, 13:1/3.0}, "risky": {11: 1/4.0, 12: 1/4.0, 13:1/4.0, 14: 1/4.0}}, 
        12: {"security": {12: 0.5, 13: 0.5}, "normal": {12: 1/3.0, 13: 1/3.0, 14:1/3.0}, "risky": {12: 1/4.0, 13: 1/4.0, 14:1/2.0}},
        13: {"security": {13: 0.5, 14: 0.5}, "normal": {13: 1/3.0, 14: 2/3.0}, "risky": {13: 1/4.0, 14: 3/4.0}},
        14: {"security": {14: 1.0}, "normal": {14: 1.0}, "risky": {14: 1.0}}
    }
    
    #take into account if we can circle
    if circle:# we can overstep state 15
        transition_prob[8]["risky"] = {8: 1/4.0, 9: 1/4.0, 14:1/4.0, 0:1/4.0}
        transition_prob[9]["risky"] = {9: 1/4.0, 14: 1/4.0, 0:1/4.0, 1:1/4.0}
        transition_prob[9]["normal"] = {9: 1/3.0, 14: 1/3.0, 0: 1/3.0}
        transition_prob[12]["risky"] = {12: 1/4.0, 13: 1/4.0, 14:1/4.0, 0: 1/4.0}
        transition_prob[13]["risky"] = {13: 1/4.0, 14: 1/4.0, 0:1/4.0, 1: 1/4.0}
        transition_prob[13]["normal"] = {13: 1/3.0, 14: 1/3.0, 0: 1/3.0}
        
    return transition_prob
 #########################################################################

 #simulation  empirical average cost
 #########################################################################
def trap(case, layout):
    """
    Gère l'effet des pièges en tenant compte des chemins slow et fast
    
    """
    freeze = False
    bonus = False
    
    if layout[case] == 1: # Restart Trap
        next_case = 0
    elif layout[case] == 2: # Penalty Trap
        if case < 3 or case == 10:
            next_case = 0
        elif case == 11 or case == 12:
            next_case = case - 10
        else:
            next_case = case - 3
    elif layout[case] == 3: # Prison Trap
        next_case = case
        freeze = True
    elif layout[case] == 4: # Bonus Trap
        next_case = case
        bonus = True # Tour gratuit
    else: # No trap
        next_case = case
        
    return next_case, freeze, bonus
    





def environment (case, dice_result, layout, circle, dice):

    freeze = False
    bonus=False
    ### DÉPLACEMENT INITIAL ###
    if dice_result == 0:
        next_case = case # Pas de déplacement
    elif case == 2: # FastLane (choix aléatoire entre deux chemins)
        next_case = 9 + dice_result if rd.randint(0, 1) else case + dice_result
    elif case in [7] and dice_result == 3: # Jump de 9 à 14
        next_case = 14
    elif case in [8] and dice_result == 2:  
        next_case = 14
    elif case in [8, 12] and dice_result == 3: # Cas cyclique pour cases 8 & 12
        next_case = 0 if circle else 14
        
    elif case in [9, 13] and dice_result == 1:  
        next_case = 14
    elif case in [9, 13]: # Cas cyclique pour cases 9 & 13
        if circle:
            if dice_result == 2:
                next_case = 0
            elif dice_result == 3:
                next_case = 1
            else:
                next_case = case + dice_result # Mouvement normal si ≠ 2 ou 3
        else:
            next_case = 14
    
    else:
        next_case = case + dice_result # Déplacement standard

    ### APPLICATION DES PIÈGES ###
    if layout[next_case] != 0:
        trap_activated = (dice == 3) or (dice == 2 and rd.randint(0, 1) == 1)
        if trap_activated:
            next_case, freeze,bonus =trap(next_case, layout)

    return next_case, freeze,bonus



def simulations_game(layout, circle, n_iteration):
    """Simule les coûts moyens de chaque case jusqu'à la destination."""
    theoretical_costs,dice_optimal = markovDecision(layout, circle) # Politique optimale
    cost = np.zeros(14) # Vecteur des coûts
    #print("Dice : ", dice_optimal)
    for pos in range(14):
        costs = [] # Liste des coûts individuels par simulation
        
        for _ in range(n_iteration):  
            case=pos
            cost_count = 0

            while case != 14: # Tant qu'on n'a pas atteint la case finale
              dice_op= dice_optimal[case] # Dé optimal
                
              if dice_op==1:
                 move= rd.choice([0,1])  
              elif dice_op== 2:
                  move= rd.choice([0,1, 2]) # Dé normal
              elif dice_op == 3:
                  move= rd.choice([0,1,2, 3]) # Grand dé risqué
                
                # Déplacement avec pièges et bonus
              next_case, freeze, bonus = environment(case, move, layout, circle,dice_op)
                
                # Gestion des coûts avec **bonus**
              cost_count += 1.0 # Tour normal
              if freeze:
                    cost_count += 1.0 # Tour supplémentaire si prison
              if bonus:
                    cost_count -= 1 # Réduction du coût si bonus

              case = next_case

            costs.append(cost_count) # Stocker le coût de cette simulation

        cost[pos] = np.mean(costs) # Calcul du coût moyen

    return cost,theoretical_costs



def plot_costs(simulated_costs, theoretical_costs):
    """
    Affiche un graphique comparant les coûts simulés et théoriques avec l'erreur relative.
    
    Paramètres :
    - simulated_costs : Coûts moyens estimés par simulation
    - theoretical_costs : Coûts optimaux obtenus par la méthode de décision de Markov
    """
    positions = np.arange(len(simulated_costs))  # Cases du plateau
    
    # Calcul de l'erreur relative (évite division par zéro)
    relative_error = np.abs((simulated_costs - theoretical_costs) / (theoretical_costs))

    # Création des sous-graphiques
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Tracé des coûts simulés et théoriques
    ax1.plot(positions, simulated_costs, marker='o', linestyle='-', color='b', label="simulated costs")
    ax1.plot(positions, theoretical_costs, marker='s', linestyle='--', color='r', label="theoretical costs")

    ax1.set_xlabel("state")
    ax1.set_ylabel("Cost")
    ax1.set_title("Comparison of simulated and theoretical costs with relative error")
    ax1.legend()
    ax1.grid(True)

    # Ajout d'un deuxième axe pour l'erreur relative
    ax2 = ax1.twinx()
    ax2.plot(positions, relative_error * 100, marker='d', linestyle=':', color='g', label="Erreur relative (%)")
    ax2.set_ylabel("Erreur relative (%)")

    # Ajout d'une légende combinée
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    
    #plt.savefig("Figure_1.pdf",format="pdf")

    plt.show()
# #################################################################

def simulations_game_with_strategy(layout, circle, n_iteration, strategy="optimal"):
    """
    Simule les coûts moyens jusqu'à la destination pour différentes stratégies.
    
    Paramètres :
    - layout : Plateau de jeu
    - circle : Règle du jeu (atterrissage exact ou non)
    - n_iteration : Nombre de simulations
    - strategy : Stratégie de choix des dés ("optimal", "dice_1", "dice_2", "dice_3", "random")
    
    """
    theoretical_costs, dice_optimal = markovDecision(layout, circle) # Politique optimale
    cost = np.zeros(14) # Vecteur des coûts

    for pos in range(14):
        costs = [] # Stockage des coûts individuels

        for _ in range(n_iteration):
            case = pos
            cost_count = 0

            while case != 14: # Tant qu'on n'a pas atteint la case finale
                
                # Choix du dé selon la stratégie
                if strategy == "optimal":
                    dice_choice = dice_optimal[case]  
                elif strategy == "dice_1":
                    dice_choice = 1  
                elif strategy == "dice_2":
                    dice_choice = 2  
                elif strategy == "dice_3":
                    dice_choice = 3  
                elif strategy == "random":
                    dice_choice = rd.choice([1, 2, 3])
                else:
                    raise ValueError("Stratégie inconnue")

                # Lancer du dé
                if dice_choice == 1:
                    move = rd.choice([0, 1])
                elif dice_choice == 2:
                    move = rd.choice([0, 1, 2])
                elif dice_choice == 3:
                    move = rd.choice([0, 1, 2, 3])

                # Déplacement et gestion des pièges
                next_case, freeze, bonus = environment(case, move, layout, circle, dice_choice)

                # Gestion des coûts
                cost_count += 1 # Coup normal
                if freeze:
                    cost_count += 1 # Tour perdu en prison
                if bonus:
                    cost_count -= 1 # Tour gratuit

                case = next_case

            costs.append(cost_count) # Stocker le coût de cette simulation

        cost[pos] = np.mean(costs) # Moyenne des coûts

    return cost, theoretical_costs

def compare_strategies(layout, circle, n_iteration):
    """
    Compare les coûts moyens pour différentes stratégies et affiche un graphique.
    """
    strategies = ["optimal", "dice_1", "dice_2", "dice_3", "random"]
    colors = ["b", "g", "r", "orange", "purple"]
    
    plt.figure(figsize=(10, 5))
    
    for strategy, color in zip(strategies, colors):
        simulated_costs, _ = simulations_game_with_strategy(layout, circle, n_iteration, strategy)
        plt.plot(np.arange(14), simulated_costs, marker='o', linestyle='-', color=color, label=strategy)

    plt.xlabel("square")
    plt.ylabel("cost")
    plt.title("Strategy comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


# #################################################################
#Bonus 
# #################################################################
def q_learning(layout, circle, episodes=100000, epsilon=0.1):
    """
    Implémente Q-learning pour le problème.
   
    - layout: configuration du plateau (15 cases)
    - circle: booléen indiquant le mode du plateau
    - episodes: nombre d'épisodes d'entraînement
    - alpha: taux d'apprentissage

    - epsilon: probabilité d'exploration 
    """
    n_states = 15  # États : 0 à 14 (14 est terminal)
    n_actions = 3  # Actions : 0 (security), 1 (normal), 2 (risky)
    
    # Initialiser la Q-table (dimension n_states x n_actions)
    Q = np.zeros((n_states, n_actions))
    N= np.zeros((n_states, n_actions))# Compteur de visites pour chaque
    def step_q(state, action):
        """
        Fonction qui simule une étape à partir d'un état et d'une action.
        Action : 0 = security, 1 = normal, 2 = risky.
        Retourne: next_state, reward, done.
        """
        # Choisir le résultat du dé en fonction de l'action
        if action == 0:  # Security
            dice_result = rd.choice([0, 1])
        elif action == 1:  # Normal
            dice_result = rd.choice([0, 1, 2])
        elif action == 2:  # Risky
            dice_result = rd.choice([0, 1, 2, 3])
        else:
            raise ValueError("Action invalide")
        
        next_state, freeze, bonus = environment(state, dice_result, layout, circle, action+1)
        
        reward = +1
        if freeze:
            reward += 1
        if bonus:
            reward -= 1
        done = (next_state == 14)
        return next_state, reward, done

    # Q-learning
    for ep in range(episodes):
        state = 0  # On commence à la case 0
        done = False
        while not done:
            #  epsilon-greedy
            
            if rd.random() < epsilon:
                action = rd.randint(0, n_actions-1)
            else:
                action = np.argmin(Q[state])
            
            next_state, reward, done = step_q(state, action)
            N[state, action] += 1
            
            # Définition du taux d'apprentissage adaptatif
            alpha = 1 / (1 + N[state, action])
            # Mise à jour Q-learning
            Q[state, action] += alpha * (reward +  np.min(Q[next_state]) - Q[state, action])
            state = next_state
            
    # Politique dérivée
    policy = np.argmin(Q, axis=1)+1
    cost_q = np.array([np.min(Q[s]) for s in range(14)])

    return cost_q, policy[:14]

# --- Fonction pour comparer Q-learning et Value Iteration ---
def compare_q_learning(layout, circle, episodes=100000):
    """
    Compare la politique apprise par Q-learning avec celle obtenue par Value Iteration.
    """
    # Q-learning
    cost_q, policy_q = q_learning(layout, circle, episodes=episodes)
    
    # Value Iteration (politique optimale)
    theoretical_costs, policy_opt = markovDecision(layout, circle)
    
    print("Politique apprise par Q-learning (1=security, 2=normal, 3=risky):")
    print(policy_q)
    print("\nPolitique optimale par Value Iteration (1=security, 2=normal, 3=risky):")
    print(policy_opt)
    
    # Visualisation simple
    positions = np.arange(0, 14)
    plt.figure(figsize=(10, 5))
    plt.plot(positions,cost_q , 'bo-', label="Q-learning cost")
    plt.plot(positions, theoretical_costs, 'rs--', label="Value Iteration cost")
    plt.xlabel(" square")
    plt.ylabel("cost")
    plt.title("Cost comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


# Exemple d'utilisation
if __name__ == "__main__":
    # Définir un layout d'exemple (15 cases)
    # 0: case normale, 1: r
    layout=np.zeros(15)
    # On teste avec circle True
    cost_empr, cost_theoretical = simulations_game(layout, False, n_iteration=10000)
    print("cost Empirique:",cost_empr)
    print("cost theoretical:",cost_theoretical)
    plot_costs(cost_empr,cost_theoretical)
    
    ##################################
    
    
    compare_strategies(layout,False,n_iteration=10000)
    compare_q_learning(layout, False, episodes=100000)
