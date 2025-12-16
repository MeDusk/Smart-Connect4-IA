import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """Architecture Dueling DQN améliorée pour Connect 4"""
    
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction avec plus de profondeur
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Avec padding=1, la taille reste 6x7
        conv_out_size = 128 * input_shape[1] * input_shape[2]
        
        # Common feature layer
        self.fc_common = nn.Linear(conv_out_size, 512)
        self.dropout = nn.Dropout(0.3)
        
        # Value stream V(s)
        self.fc_value = nn.Linear(512, 256)
        self.value_out = nn.Linear(256, 1)
        
        # Advantage stream A(s,a)
        self.fc_advantage = nn.Linear(512, 256)
        self.advantage_out = nn.Linear(256, num_actions)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation Kaiming He"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Common features
        x = F.relu(self.fc_common(x))
        x = self.dropout(x)
        
        # Dueling streams
        value = F.relu(self.fc_value(x))
        value = self.value_out(value)
        
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage_out(advantage)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# Exemple d'utilisation 
if __name__ == '__main__':
    # Définition des paramètres
    INPUT_SHAPE = (3, 6, 7)
    NUM_ACTIONS = 7
    
    # Création du modèle
    model = DuelingDQN(INPUT_SHAPE, NUM_ACTIONS)
    
    # Création d'un tenseur d'entrée factice (Batch size = 1)
    dummy_input = torch.randn(1, *INPUT_SHAPE)
    
    # Passage avant
    q_values = model(dummy_input)
    
    # Affichage des résultats
    print(f"Forme de l'entrée: {dummy_input.shape}")
    print(f"Forme de la sortie (Q-values): {q_values.shape}")
    print(f"Q-values pour les 7 actions: {q_values.detach().numpy()}")
    
    # Vérification que le modèle est sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    model.to(device)
    
    # Test avec un batch plus grand
    batch_input = torch.randn(16, *INPUT_SHAPE).to(device)
    batch_q_values = model(batch_input)
    print(f"Forme de la sortie avec un batch de 16: {batch_q_values.shape}")
