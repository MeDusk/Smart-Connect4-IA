import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_rewards(file_path='training_rewards.npy', window=100):
    """
    Charge les récompenses d'entraînement et trace la moyenne glissante.
    """
    if not os.path.exists(file_path):
        print(f"Fichier de récompenses non trouvé: {file_path}")
        return

    rewards = np.load(file_path)
    
    # Calcul de la moyenne glissante
    # Utilisation de 'valid' pour s'assurer que la fenêtre est complète
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(moving_avg)), moving_avg)
    plt.title(f'Performance d\'Entraînement du DQN (Moyenne Glissante sur {window} Épisodes)')
    plt.xlabel('Épisode')
    plt.ylabel(f'Récompense Moyenne (Fenêtre de {window})')
    plt.grid(True)
    plt.savefig('training_performance.png')
    plt.close()
    print("Graphique de performance d'entraînement sauvegardé: training_performance.png")

def plot_test_results(file_path='test_results.npy'):
    """
    Charge les résultats de test et trace un graphique en barres des taux de victoire/défaite/nul.
    """
    if not os.path.exists(file_path):
        print(f"Fichier de résultats de test non trouvé: {file_path}")
        return

    results = np.load(file_path)
    
    win_count = np.sum(results == 'Win')
    loss_count = np.sum(results == 'Loss')
    draw_count = np.sum(results == 'Draw')
    total = len(results)
    
    win_rate = win_count / total
    loss_rate = loss_count / total
    draw_rate = draw_count / total
    
    labels = ['Victoire (IA)', 'Défaite (IA)', 'Match Nul']
    rates = [win_rate, loss_rate, draw_rate]
    
    plt.figure(figsize=(8, 8))
    plt.pie(rates, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336', '#FFC107'])
    plt.title(f'Résultats des Tests du DQN contre Adversaire Aléatoire ({total} Parties)')
    plt.savefig('test_results_pie.png')
    plt.close()
    print("Graphique des résultats de test sauvegardé: test_results_pie.png")

if __name__ == '__main__':
    try:
        plot_training_rewards()
        plot_test_results()
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {e}")
        print("Assurez-vous que les fichiers 'training_rewards.npy' et 'test_results.npy' existent.")
