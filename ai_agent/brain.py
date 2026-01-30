import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ROWS = 6
COLS = 7

# --- 1. DEFINITION DU MODELE ---
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(ROWS * COLS, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Branche Value
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Branche Advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, COLS)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean())

# --- 2. CHARGEMENT DU CERVEAU ---
model = DuelingDQN()

try:
    # On charge le fichier généré par le script d'entraînement
    model.load_state_dict(torch.load("models/smart_dqn.pth", map_location=torch.device('cpu')))
    model.eval() # Mode évaluation
    print("Cerveau Dueling DQN chargé avec succès !")
except Exception as e:
    print(f"ERREUR : Impossible de charger 'smart_dqn.pth'.\nErreur: {e}")

# --- 3. FONCTIONS REFLEXES ---
def check_win_simulated(board, piece):
    # Vérifie si 'piece' a gagné sur le plateau donné
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Diagonales
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
            if board[r+3][c] == piece and board[r+2][c+1] == piece and board[r+1][c+2] == piece and board[r][c+3] == piece:
                return True
    return False

def find_immediate_threat(board, piece_to_check):
    # Simule chaque coup possible pour voir s'il mène à une victoire immédiate
    valid_moves = [c for c in range(COLS) if board[0][c] == 0]
    
    for col in valid_moves:
        temp_board = board.copy()
        # Faire tomber la pièce virtuellement
        for r in range(ROWS-1, -1, -1):
            if temp_board[r][col] == 0:
                temp_board[r][col] = piece_to_check
                break
        
        if check_win_simulated(temp_board, piece_to_check):
            return col # On a trouvé le coup critique !
    return None

# --- 4. API ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Node envoie : 1=Humain, -1=IA, 0=Vide
        board = np.array(data['board'])

        # --- PHASE 1 : REFLEXES (Gagner ou Bloquer) ---
        
        # A. Est-ce que l'IA (-1) peut gagner MAINTENANT ?
        win_move = find_immediate_threat(board, -1)
        if win_move is not None:
            print(f"COUP GAGNANT DÉTECTÉ : Colonne {win_move}")
            return jsonify({"column": int(win_move)})

        # B. Est-ce que l'Humain (1) va gagner au prochain tour ? (BLOCAGE)
        block_move = find_immediate_threat(board, 1)
        if block_move is not None:
            print(f"BLOCAGE D'URGENCE : Colonne {block_move}")
            return jsonify({"column": int(block_move)})

        # --- PHASE 2 : REFLEXION DEEP LEARNING (Si pas d'urgence) ---
        
        # Inversion : L'IA a appris en étant le joueur "1".
        # Donc on multiplie par -1 pour qu'elle voit ses pions (-1) comme des 1.
        ai_vision = board * -1
        flat_board = ai_vision.flatten()

        with torch.no_grad():
            tensor_board = torch.FloatTensor(flat_board).unsqueeze(0)
            
            # Le Dueling DQN renvoie des Q-Values (Scores), pas des probas
            q_values = model(tensor_board).numpy()[0]
            
            # On filtre les coups impossibles (colonnes pleines)
            valid_moves = [c for c in range(COLS) if board[0][c] == 0]
            
            if not valid_moves:
                return jsonify({"error": "Game Over"}), 400
            
            # On met une valeur très basse (-9999) aux colonnes pleines
            for i in range(COLS):
                if i not in valid_moves:
                    q_values[i] = -99999.0
            
            # On prend la colonne avec le meilleur score Q
            best_col = int(np.argmax(q_values))
            
            print(f"Stratégie DQN : Colonne {best_col} (Score: {q_values[best_col]:.2f})")
            return jsonify({"column": best_col})

    except Exception as e:
        print("Erreur:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Cerveau Dueling DQN (Mode Hybride) prêt sur le port 5000...")
    app.run(host='0.0.0.0', port=5000)