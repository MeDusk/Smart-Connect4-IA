import torch
import numpy as np
from tqdm import tqdm
from .connect_four_env import ConnectFourEnv
from .dqn_agent import DQNAgent

def train_dqn_self_play(num_episodes=15000):
    """Entraînement en self-play amélioré"""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    
    env = ConnectFourEnv()
    
    # Agent principal
    agent = DQNAgent(
        input_shape=(3, 6, 7),
        num_actions=7,
        device=DEVICE,
        GAMMA=0.99,
        LR=0.0001,
        BATCH_SIZE=64,
        TARGET_UPDATE=500,
        BUFFER_CAPACITY=100000,
        EPS_START=1.0,
        EPS_END=0.01,
        EPS_DECAY=20000
    )
    
    # Clone comme adversaire
    opponent = DQNAgent(
        input_shape=(3, 6, 7),
        num_actions=7,
        device=DEVICE,
        GAMMA=0.99,
        LR=0.0001,
        EPS_START=0.1,
        EPS_END=0.01,
        EPS_DECAY=10000
    )
    
    episode_rewards = []
    wins, losses, draws = 0, 0, 0
    
    pbar = tqdm(range(num_episodes))
    for episode in pbar:
        state = env.reset()
        done = False
        total_reward = 0
        
        # Alterne qui commence
        is_agent_turn = (episode % 2 == 0)
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            if is_agent_turn:
                action_tensor = agent.select_action(state_tensor, valid_actions)
                action = action_tensor.item()
            else:
                action_tensor = opponent.select_action(state_tensor, valid_actions)
                action = action_tensor.item()
            
            next_state, reward, done, info = env.step(action)
            
            # Stocker et optimiser pour l'agent principal uniquement
            if is_agent_turn:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
                agent.memory.push(state_tensor, action_tensor, reward, 
                                next_state_tensor, done)
                loss = agent.optimize_model()
                total_reward += reward
            
            state = next_state
            is_agent_turn = not is_agent_turn
        
        # Statistiques
        if env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        else:
            draws += 1
        
        episode_rewards.append(total_reward)
        
        # Mise à jour adversaire tous les 100 épisodes
        if episode % 100 == 0 and episode > 0:
            opponent.policy_net.load_state_dict(agent.policy_net.state_dict())
            opponent.target_net.load_state_dict(agent.target_net.state_dict())
            
            win_rate = wins / 100
            pbar.set_postfix({
                'WinRate': f'{win_rate:.2%}',
                'Eps': f'{agent.epsilon:.3f}',
                'AvgReward': f'{np.mean(episode_rewards[-100:]):.2f}'
            })
            wins = losses = draws = 0
        
        # Mise à jour target network
        if episode % agent.TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Sauvegarde
        if episode % 2000 == 0 and episode > 0:
            agent.save_model(f"models/dqn_episode_{episode}.pth")
            print(f"\nModèle sauvegardé à l'épisode {episode}")
    
    # Sauvegarde finale
    agent.save_model("models/dqn_connect_four_final.pth")
    np.save('training_rewards.npy', episode_rewards)
    print("\nEntraînement terminé!")
    
    return agent

if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    train_dqn_self_play(num_episodes=15000)
