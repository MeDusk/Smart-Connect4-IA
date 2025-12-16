# test_dqn.py
import torch
import numpy as np
import random
from .connect_four_env import ConnectFourEnv
from .dqn_agent import DQNAgent

#  Configuration de Test 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SHAPE = (3, 6, 7)
NUM_ACTIONS = 7
MODEL_PATH = "models/dqn_connect_four_final.pth"  
NUM_TEST_GAMES = 100
RENDER_GAME = True  

def random_opponent_action(env):
    """Adversaire aléatoire simple"""
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None
    return random.choice(valid_actions)

def test_dqn(num_games=100, render_first=True, model_path=MODEL_PATH):
    """
    Évalue l'agent DQN entraîné contre un adversaire aléatoire
    """
    print(f"Chargement du modèle depuis {model_path}...")
    
    # Initialiser l'environnement et l'agent
    env = ConnectFourEnv()
    agent = DQNAgent(
        input_shape=INPUT_SHAPE,
        num_actions=NUM_ACTIONS,
        device=DEVICE,
        EPS_START=0.0,  # Pas d'exploration pendant les tests
        EPS_END=0.0
    )
    
    # Charger le modèle entraîné
    try:
        agent.load_model(model_path)
        print(" Modèle chargé avec succès!")
    except FileNotFoundError:
        print(f" Erreur: Modèle non trouvé à {model_path}")
        print("Entraînez d'abord le modèle avec: python train_dqn.py")
        return None, None
    
    # Statistiques
    wins = 0
    losses = 0
    draws = 0
    game_histories = []
    
    print(f"\n🎮 Démarrage des tests sur {num_games} parties...\n")
    
    for game_num in range(1, num_games + 1):
        state = env.reset()
        done = False
        game_history = []
        
        # L'agent (joueur 1) commence toujours
        env.current_player = 1
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            #  Tour de l'Agent (Joueur 1) 
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action_tensor = agent.select_action(state_tensor, valid_actions)
            action = action_tensor.item()
            
            game_history.append(('Agent', action))
            next_state, reward, done, info = env.step(action)
            
            if done:
                break
            
            #  Tour de l'Adversaire (Joueur 2) 
            env.current_player = 2
            opponent_action = random_opponent_action(env)
            
            if opponent_action is not None:
                game_history.append(('Opponent', opponent_action))
                next_state, reward, done, info = env.step(opponent_action)
            else:
                done = True  # Match nul si adversaire ne peut pas jouer
                break
            
            env.current_player = 1
            state = next_state
        
        # Comptabiliser les résultats
        if env.winner == 1:
            wins += 1
            result = "Victoire"
        elif env.winner == 2:
            losses += 1
            result = "Défaite"
        else:
            draws += 1
            result = "Match Nul"
        
        game_histories.append((game_history, result))
        
        # Afficher progression
        if game_num % 10 == 0:
            print(f"Partie {game_num}/{num_games} - {result}")
    
    # Afficher la première partie si demandé
    if render_first and len(game_histories) > 0:
        print("\n" + "="*50)
        print("VISUALISATION DE LA PREMIÈRE PARTIE")
        print("="*50)
        env.reset()
        for player, action in game_histories[0][0]:
            env.current_player = 1 if player == 'Agent' else 2
            env.step(action)
        env.render()
        print(f"Résultat: {game_histories[0][1]}")
        print("="*50 + "\n")
    
    #Affichage des Statistiques Finales
    total_games = num_games
    win_rate = (wins / total_games) * 100
    loss_rate = (losses / total_games) * 100
    draw_rate = (draws / total_games) * 100
    
    print("\n" + "="*50)
    print("RÉSULTATS DES TESTS")
    print("="*50)
    print(f"Total des parties jouées : {total_games}")
    print(f"Victoires de l'Agent     : {wins:3d} ({win_rate:5.1f}%)")
    print(f"Défaites de l'Agent      : {losses:3d} ({loss_rate:5.1f}%)")
    print(f"Matchs nuls              : {draws:3d} ({draw_rate:5.1f}%)")
    print("="*50)
    
    # Évaluation de la performance
    if win_rate >= 80:
        print("Performance EXCELLENTE!")
    elif win_rate >= 60:
        print("Performance BONNE")
    elif win_rate >= 40:
        print("Performance MOYENNE")
    else:
        print("Performance FAIBLE - Entraînement supplémentaire requis")
    
    print("="*50 + "\n")
    
    # Sauvegarder les résultats
    results = np.array([1 if r == "Victoire" else (0 if r == "Match Nul" else -1) 
                       for _, r in game_histories])
    np.save('test_results.npy', results)
    print("Résultats sauvegardés dans 'test_results.npy'")
    
    return game_histories, win_rate

if __name__ == '__main__':
    import os
    
    # Cherche le dernier modèle sauvegardé
    model_files = []
    if os.path.exists('models'):
        model_files = [f"models/{f}" for f in os.listdir('models') 
                      if f.endswith('.pth')]
    
    if not model_files:
        print("Aucun modèle trouvé dans le dossier 'models/'")
        print("Entraînez d'abord avec: python train_dqn.py")
    else:
        # Utiliser le dernier modèle 
        if any('final' in f for f in model_files):
            model_path = [f for f in model_files if 'final' in f][0]
        else:
            model_path = sorted(model_files)[-1]
        
        print(f"Utilisation du modèle: {model_path}\n")
        
        # Lancer les tests
        results, win_rate = test_dqn(
            num_games=NUM_TEST_GAMES,
            render_first=RENDER_GAME,
            model_path=model_path
        )
