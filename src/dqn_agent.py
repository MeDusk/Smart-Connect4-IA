import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import numpy as np

# Import du modèle DQN
from .dqn_model import DuelingDQN

# Définition d'une transition pour le Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    Mémoire de relecture (Replay Buffer) pour stocker les expériences (transitions).
    Essentiel pour le DQN afin de briser la corrélation entre les échantillons 
    séquentiels et d'améliorer la stabilité de l'apprentissage.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """ Ajoute une transition à la mémoire. """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Sélectionne un échantillon aléatoire de transitions. """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """ Retourne la taille actuelle de la mémoire. """
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, *args):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon
    
    def __len__(self):
        return self.size


class DQNAgent:
    """
    Agent Deep Q-Network (DQN) pour le jeu de Puissance 4.
    Gère la sélection d'action, l'apprentissage et la mise à jour du réseau cible.
    """
    def __init__(self, input_shape, num_actions, device, **kwargs):
        """
        Initialisation de l'agent DQN.
        
        :param input_shape: Forme de l'état (3, 6, 7).
        :param num_actions: Nombre d'actions (7).
        :param device: 'cuda' ou 'cpu'.
        :param kwargs: Hyperparamètres (GAMMA, LR, BUFFER_CAPACITY, etc.).
        """
        self.device = device
        self.num_actions = num_actions
        
        # Hyperparamètres
        self.GAMMA = kwargs.get('GAMMA', 0.99)
        self.LR = kwargs.get('LR', 1e-4)
        self.BATCH_SIZE = kwargs.get('BATCH_SIZE', 64)
        self.TARGET_UPDATE = kwargs.get('TARGET_UPDATE', 100)
        self.EPS_START = kwargs.get('EPS_START', 1.0)
        self.EPS_END = kwargs.get('EPS_END', 0.01)
        self.EPS_DECAY = kwargs.get('EPS_DECAY', 10000)
        self.epsilon = self.EPS_START  # Initialisation de epsilon
        # Réseaux
        self.policy_net = DuelingDQN(input_shape, num_actions).to(device)
        self.target_net = DuelingDQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Le réseau cible n'est pas entraîné
        
        # Optimiseur et Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.criterion = nn.SmoothL1Loss() # Huber Loss, plus robuste que MSE
        
        # Mémoire de relecture
        self.memory = PrioritizedReplayBuffer(kwargs.get('BUFFER_CAPACITY', 100000))
        
        self.steps_done = 0

    def select_action(self, state, valid_actions):
        """
        Sélectionne une action en utilisant la stratégie epsilon-greedy.
        
        :param state: État actuel du jeu (Tensor).
        :param valid_actions: Liste des indices de colonnes jouables.
        :return: Action sélectionnée (Tensor scalaire).
        """
        # Calcul de epsilon pour l'exploration
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.epsilon = eps_threshold
        self.steps_done += 1
        
        if random.random() < eps_threshold:
            # Exploration: Choisir une action aléatoire parmi les actions valides
            action = random.choice(valid_actions)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
        else:
            # Exploitation: Choisir l'action avec la Q-value maximale
            with torch.no_grad():
                # Le réseau retourne les Q-values pour toutes les actions
                q_values = self.policy_net(state)
                
                # Masquer les actions invalides en leur donnant une très faible Q-value
                # (pour s'assurer qu'elles ne sont pas choisies)
                mask = torch.full_like(q_values, -float('inf'))
                for action_idx in valid_actions:
                    mask[0, action_idx] = 0
                
                masked_q_values = q_values + mask
                
                # Sélectionner l'action avec la Q-value maximale
                action = masked_q_values.argmax(1).item()
                return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Optimisation avec Double DQN et Prioritized Replay"""
        if len(self.memory) < self.BATCH_SIZE:
            return None
    
    # Sample avec priorités
        transitions, indices, weights = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        weights_batch = torch.FloatTensor(weights).to(self.device)
    
    # Q-values actuelles
        current_q = self.policy_net(state_batch).gather(1, action_batch)
    
    # Double DQN: sélection avec policy_net, évaluation avec target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            next_q = next_q * (1 - done_batch)
            target_q = reward_batch + self.GAMMA * next_q
    
    # TD-errors pour mise à jour des priorités
        td_errors = (current_q.squeeze(1) - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
    
    # Loss pondérée
        loss = (weights_batch * self.criterion(
            current_q.squeeze(1), 
            target_q
        )).mean()
    
    # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
    
        return loss.item()


    def update_target_network(self):
        """
        Met à jour le réseau cible en copiant les poids du réseau de politique.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """ Sauvegarde les poids du réseau de politique. """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """ Charge les poids du réseau de politique. """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval() 
        self.target_net.eval()
        
# Exemple d'utilisation 
if __name__ == '__main__':
    # Nécessite d'importer l'environnement pour un test complet
    from connect_four_env import ConnectFourEnv
    
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SHAPE = (3, 6, 7)
    NUM_ACTIONS = 7
    
    # Initialisation
    env = ConnectFourEnv()
    agent = DQNAgent(INPUT_SHAPE, NUM_ACTIONS, DEVICE)
    
    print(f"Agent initialisé sur le device: {DEVICE}")
    print(f"Taille de la mémoire de relecture: {len(agent.memory)}")
    
    # Simulation d'une transition
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    valid_actions = env.get_valid_actions()
    action_tensor = agent.select_action(state_tensor, valid_actions)
    action = action_tensor.item()
    
    next_state, reward, done, info = env.step(action)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Enregistrement de la transition
    agent.memory.push(state_tensor, action_tensor, reward, next_state_tensor, done)
    print(f"Taille de la mémoire après push: {len(agent.memory)}")
    
    # Test de l'optimisation (nécessite plus d'échantillons)
    print("Test d'optimisation (nécessite BATCH_SIZE échantillons)...")
    # Pour un test réel, il faudrait remplir la mémoire jusqu'à BATCH_SIZE
    # Ici, on se contente de vérifier que la méthode existe et ne plante pas immédiatement.
    # loss = agent.optimize_model()
    # print(f"Loss: {loss}")
    
    # Test de la mise à jour du réseau cible
    agent.update_target_network()
    print("Réseau cible mis à jour.")
