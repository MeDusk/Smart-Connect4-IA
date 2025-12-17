# #!/usr/bin/env python3
# """

# SMART-CONNECT4 - MODULE D'INFÉRENCE IA    
#   Script d'interface entre l'agent DQN et le backend Node.js                  
#   Communication : StdIO/JSON                                                  
#   Protocole : Request/Response avec gestion d'erreurs robuste 
#   Auteur : Mohamed NAJID                                                      
#   Projet : Smart-Connect4 - M2 IA UCBL                                                                                              

# """

# import sys
# import json
# import time
# import logging
# from pathlib import Path
# from datetime import datetime

# import torch
# import numpy as np

# # Import des modules du projet
# try:
#     from src.dqn_agent import DQNAgent
#     from src.dqn_model import DuelingDQN
# except ImportError:
#     sys.path.insert(0, str(Path(__file__).parent))
#     from dqn_agent import DQNAgent
#     from dqn_model import DuelingDQN


# # ═══════════════════════════════════════════════════════════════════════════
# #                          CONFIGURATION & LOGGING

# class Config:
#     """Configuration du module d'inférence"""
    
#     # Chemins
#     MODEL_PATH = "models/dqn_connect_four_final.pth"
#     LOG_FILE = "logs/inference.log"
    
#     # Paramètres du jeu
#     INPUT_SHAPE = (3, 6, 7)
#     NUM_ACTIONS = 7
#     ROWS = 6
#     COLS = 7
    
#     # Optimisation
#     USE_CPU = True  
#     TIMEOUT_PREDICTION = 5.0  # secondes
    
#     # Logging
#     LOG_LEVEL = logging.INFO
#     LOG_TO_FILE = False  # Désactiver en production pour éviter I/O


# def setup_logging():
#     """Configure le système de logging"""
#     log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    
#     handlers = [logging.StreamHandler(sys.stderr)]
    
#     if Config.LOG_TO_FILE:
#         Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
#         handlers.append(logging.FileHandler(Config.LOG_FILE))
    
#     logging.basicConfig(
#         level=Config.LOG_LEVEL,
#         format=log_format,
#         handlers=handlers
#     )
    
#     return logging.getLogger(__name__)


# logger = setup_logging()


# # GESTION DU PLATEAU

# class BoardManager:
#     """Gestionnaire du plateau de jeu avec validation"""
    
#     @staticmethod
#     def validate_board(board):
#         """
#         Valide la structure et le contenu du plateau
        
#         Args:
#             board: Liste 2D représentant le plateau
            
#         Returns:
#             tuple: (is_valid, error_message)
#         """
#         if not isinstance(board, list):
#             return False, "Board must be a list"
        
#         if len(board) != Config.ROWS:
#             return False, f"Board must have {Config.ROWS} rows, got {len(board)}"
        
#         for i, row in enumerate(board):
#             if not isinstance(row, list):
#                 return False, f"Row {i} must be a list"
            
#             if len(row) != Config.COLS:
#                 return False, f"Row {i} must have {Config.COLS} columns, got {len(row)}"
            
#             for j, cell in enumerate(row):
#                 if cell not in [0, 1, 2]:
#                     return False, f"Invalid cell value at ({i},{j}): {cell}. Must be 0, 1, or 2"
        
#         return True, None
    
#     @staticmethod
#     def board_to_state(board):
#         """
#         Convertit le plateau JSON en tenseur (3, 6, 7) pour le réseau
        
#         Args:
#             board: Liste 2D (6x7) avec valeurs 0 (vide), 1 (joueur 1), 2 (joueur 2)
            
#         Returns:
#             torch.Tensor: Tenseur (1, 3, 6, 7) prêt pour l'inférence
#         """
#         board_array = np.array(board, dtype=np.float32)
        
#         # Créer les 3 canaux
#         state = np.zeros((3, Config.ROWS, Config.COLS), dtype=np.float32)
#         state[0] = (board_array == 1).astype(np.float32)  # Joueur 1
#         state[1] = (board_array == 2).astype(np.float32)  # Joueur 2
#         state[2] = (board_array == 0).astype(np.float32)  # Cases vides
        
#         # Ajouter dimension batch
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
#         return state_tensor
    
#     @staticmethod
#     def get_valid_actions(board):
#         """
#         Retourne les colonnes jouables (non pleines)
        
#         Args:
#             board: Liste 2D représentant le plateau
            
#         Returns:
#             list: Indices des colonnes jouables (0-6)
#         """
#         return [col for col in range(Config.COLS) if board[0][col] == 0]
    
#     @staticmethod
#     def get_board_stats(board):
#         """Statistiques du plateau pour monitoring"""
#         board_array = np.array(board)
#         return {
#             'player1_pieces': int(np.sum(board_array == 1)),
#             'player2_pieces': int(np.sum(board_array == 2)),
#             'empty_cells': int(np.sum(board_array == 0)),
#             'valid_actions': len(BoardManager.get_valid_actions(board))
#         }



# #                          PONT IA (AI BRIDGE)


# class AIBridge:
#     """
#     Pont entre le système Node.js et l'agent DQN
#     Gère le chargement du modèle et les prédictions
#     """
    
#     def __init__(self, model_path=None):
#         """
#         Initialise le pont IA
        
#         Args:
#             model_path: Chemin vers le modèle PyTorch (.pth)
#         """
#         self.model_path = model_path or Config.MODEL_PATH
#         self.device = self._setup_device()
#         self.agent = None
#         self.load_time = None
#         self.prediction_count = 0
#         self.total_inference_time = 0.0
        
#         logger.info("="*70)
#         logger.info("INITIALISATION DU MODULE IA")
#         logger.info("="*70)
#         logger.info(f"Device sélectionné: {self.device}")
#         logger.info(f"Modèle: {self.model_path}")
        
#         self._load_model()
    
#     def _setup_device(self):
#         """Configure le device (CPU/GPU) optimal"""
#         if Config.USE_CPU:
#             device = torch.device("cpu")
#             logger.info("Mode CPU forcé pour latence prévisible")
#         else:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             if device.type == "cuda":
#                 logger.info(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        
#         return device
    
#     def _load_model(self):
#         """Charge le modèle DQN entraîné"""
#         start_time = time.time()
        
#         try:
#             # Vérifier que le fichier existe
#             if not Path(self.model_path).exists():
#                 raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
            
#             logger.info(f"Chargement du modèle depuis {self.model_path}...")
            
#             # Créer l'agent
#             self.agent = DQNAgent(
#                 input_shape=Config.INPUT_SHAPE,
#                 num_actions=Config.NUM_ACTIONS,
#                 device=self.device
#             )
            
#             # Charger les poids
#             self.agent.load_model(self.model_path)
            
#             # Mode évaluation (désactive dropout, batchnorm en mode eval)
#             self.agent.policy_net.eval()
#             self.agent.epsilon = 0.0  # Pas d'exploration en production
            
#             self.load_time = time.time() - start_time
            
#             logger.info(f"✓ Modèle chargé avec succès en {self.load_time:.3f}s")
#             logger.info(f"  Architecture: {self.agent.policy_net.__class__.__name__}")
#             logger.info(f"  Paramètres: {sum(p.numel() for p in self.agent.policy_net.parameters()):,}")
            
#         except Exception as e:
#             logger.error(f"✗ Erreur lors du chargement du modèle: {e}")
#             raise
    
#     def predict(self, board):
#         """
#         Prédit le meilleur coup à jouer
        
#         Args:
#             board: Plateau de jeu (liste 2D)
            
#         Returns:
#             dict: {"column": int, "confidence": float, "q_values": list, "inference_time_ms": float}
            
#         Raises:
#             ValueError: Si le plateau est invalide ou aucune action valide
#         """
#         start_time = time.time()
        
#         try:
#             # 1. Validation du plateau
#             is_valid, error_msg = BoardManager.validate_board(board)
#             if not is_valid:
#                 raise ValueError(f"Plateau invalide: {error_msg}")
            
#             # 2. Obtenir les actions valides
#             valid_actions = BoardManager.get_valid_actions(board)
#             if not valid_actions:
#                 raise ValueError("Aucune action valide: plateau plein")
            
#             # 3. Conversion en tenseur
#             state_tensor = BoardManager.board_to_state(board).to(self.device)
            
#             # 4. Inférence
#             with torch.no_grad():
#                 # Obtenir les Q-values brutes
#                 q_values = self.agent.policy_net(state_tensor).squeeze(0)
                
#                 # Masquer les actions invalides
#                 mask = torch.full_like(q_values, -float('inf'))
#                 for action_idx in valid_actions:
#                     mask[action_idx] = 0
#                 masked_q_values = q_values + mask
                
#                 # Sélectionner la meilleure action
#                 best_action = masked_q_values.argmax().item()
#                 confidence = torch.softmax(masked_q_values[valid_actions], dim=0).max().item()
            
#             # 5. Métriques
#             inference_time = (time.time() - start_time) * 1000  # ms
#             self.prediction_count += 1
#             self.total_inference_time += inference_time
            
#             # 6. Logging détaillé
#             stats = BoardManager.get_board_stats(board)
#             logger.info(f"Prédiction #{self.prediction_count}")
#             logger.info(f"  État: J1={stats['player1_pieces']} | J2={stats['player2_pieces']} | Vide={stats['empty_cells']}")
#             logger.info(f"  Actions valides: {valid_actions}")
#             logger.info(f"  Coup choisi: Colonne {best_action}")
#             logger.info(f"  Confiance: {confidence:.1%}")
#             logger.info(f"  Temps: {inference_time:.2f}ms")
            
#             # 7. Retourner le résultat complet
#             return {
#                 "column": best_action,
#                 "confidence": round(confidence, 4),
#                 "q_values": [round(float(q), 3) for q in q_values.cpu().tolist()],
#                 "valid_actions": valid_actions,
#                 "inference_time_ms": round(inference_time, 2),
#                 "board_stats": stats
#             }
            
#         except Exception as e:
#             logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
#             raise
    
#     def get_stats(self):
#         """Retourne les statistiques d'utilisation"""
#         avg_time = self.total_inference_time / self.prediction_count if self.prediction_count > 0 else 0
#         return {
#             "predictions_count": self.prediction_count,
#             "average_inference_time_ms": round(avg_time, 2),
#             "total_inference_time_s": round(self.total_inference_time / 1000, 2),
#             "model_load_time_s": round(self.load_time, 3) if self.load_time else None,
#             "device": str(self.device)
#         }



# #                       GESTIONNAIRE DE COMMUNICATION


# class CommunicationHandler:
#     """
#     Gestionnaire de communication StdIO avec Node.js
#     Protocole: JSON lines (une requête par ligne, une réponse par ligne)
#     """
    
#     def __init__(self, ai_bridge):
#         """
#         Args:
#             ai_bridge: Instance de AIBridge
#         """
#         self.ai = ai_bridge
#         self.request_count = 0
#         self.start_time = time.time()
        
#         logger.info("Gestionnaire de communication initialisé")
    
#     def send_response(self, response):
#         """
#         Envoie une réponse JSON sur stdout
        
#         Args:
#             response: Dictionnaire à sérialiser en JSON
#         """
#         try:
#             json_response = json.dumps(response)
#             print(json_response, flush=True)  # flush=True crucial pour StdIO
#         except Exception as e:
#             logger.error(f"Erreur lors de l'envoi de la réponse: {e}")
#             # Envoyer une erreur basique
#             print(json.dumps({"status": "error", "message": "Response serialization failed"}), flush=True)
    
#     def handle_predict(self, request):
#         """
#         Gère une requête de prédiction
        
#         Args:
#             request: Dictionnaire contenant la clé "board"
            
#         Returns:
#             dict: Réponse formatée
#         """
#         try:
#             # Extraire le plateau
#             board = request.get("board")
#             if board is None:
#                 return {
#                     "status": "error",
#                     "error_code": "MISSING_BOARD",
#                     "message": "Missing 'board' field in request"
#                 }
            
#             # Prédire
#             result = self.ai.predict(board)
            
#             # Formater la réponse
#             return {
#                 "status": "success",
#                 "column": result["column"],
#                 "metadata": {
#                     "confidence": result["confidence"],
#                     "inference_time_ms": result["inference_time_ms"],
#                     "valid_actions": result["valid_actions"],
#                     "board_stats": result["board_stats"],
#                     "timestamp": datetime.now().isoformat()
#                 },
#                 # Q-values optionnelles (utiles pour debugging côté Node.js)
#                 "debug": {
#                     "q_values": result["q_values"]
#                 } if logger.level == logging.DEBUG else {}
#             }
            
#         except ValueError as e:
#             return {
#                 "status": "error",
#                 "error_code": "INVALID_BOARD",
#                 "message": str(e)
#             }
#         except Exception as e:
#             logger.error(f"Erreur inattendue dans handle_predict: {e}", exc_info=True)
#             return {
#                 "status": "error",
#                 "error_code": "PREDICTION_FAILED",
#                 "message": f"Unexpected error: {str(e)}"
#             }
    
#     def handle_ping(self, request):
#         """Gère une requête de ping (health check)"""
#         uptime = time.time() - self.start_time
#         return {
#             "status": "pong",
#             "uptime_seconds": round(uptime, 2),
#             "stats": self.ai.get_stats()
#         }
    
#     def handle_stats(self, request):
#         """Retourne les statistiques détaillées"""
#         return {
#             "status": "success",
#             "stats": self.ai.get_stats(),
#             "uptime_seconds": round(time.time() - self.start_time, 2),
#             "total_requests": self.request_count
#         }
    
#     def handle_request(self, line):
#         """
#         Traite une ligne de requête JSON
        
#         Args:
#             line: Ligne JSON contenant la requête
            
#         Returns:
#             dict: Réponse à envoyer
#         """
#         self.request_count += 1
        
#         try:
#             # Parser le JSON
#             request = json.loads(line.strip())
#             command = request.get("command", "predict")  # "predict" par défaut
            
#             logger.debug(f"Requête #{self.request_count}: {command}")
            
#             # Router la commande
#             if command == "predict":
#                 return self.handle_predict(request)
#             elif command == "ping":
#                 return self.handle_ping(request)
#             elif command == "stats":
#                 return self.handle_stats(request)
#             elif command == "shutdown":
#                 logger.info("Commande de shutdown reçue")
#                 return {"status": "shutdown", "message": "Shutting down gracefully"}
#             else:
#                 return {
#                     "status": "error",
#                     "error_code": "UNKNOWN_COMMAND",
#                     "message": f"Unknown command: {command}"
#                 }
                
#         except json.JSONDecodeError as e:
#             logger.error(f"Erreur de parsing JSON: {e}")
#             return {
#                 "status": "error",
#                 "error_code": "INVALID_JSON",
#                 "message": f"JSON parsing error: {str(e)}"
#             }
#         except Exception as e:
#             logger.error(f"Erreur inattendue: {e}", exc_info=True)
#             return {
#                 "status": "error",
#                 "error_code": "INTERNAL_ERROR",
#                 "message": f"Internal error: {str(e)}"
#             }
    
#     def run(self):
#         """
#         Boucle principale d'écoute StdIO
#         Écoute stdin ligne par ligne et répond sur stdout
#         """
#         logger.info("="*70)
#         logger.info("SERVEUR D'INFÉRENCE DÉMARRÉ")
#         logger.info("="*70)
#         logger.info("En attente de requêtes sur stdin...")
        
#         # Signal de readiness pour Node.js
#         self.send_response({
#             "status": "ready",
#             "message": "AI inference server ready",
#             "config": {
#                 "model": Config.MODEL_PATH,
#                 "device": str(self.ai.device),
#                 "version": "1.0.0"
#             }
#         })
        
#         try:
#             # Boucle d'écoute infinie
#             for line in sys.stdin:
#                 if not line.strip():
#                     continue  # Ignorer les lignes vides
                
#                 # Traiter la requête
#                 response = self.handle_request(line)
                
#                 # Envoyer la réponse
#                 self.send_response(response)
                
#                 # Arrêt propre si shutdown demandé
#                 if response.get("status") == "shutdown":
#                     break
            
#             logger.info("Sortie de la boucle d'écoute (stdin fermé)")
            
#         except KeyboardInterrupt:
#             logger.info("Interruption clavier détectée (Ctrl+C)")
#         except Exception as e:
#             logger.error(f"Erreur fatale dans la boucle principale: {e}", exc_info=True)
#             self.send_response({
#                 "status": "fatal_error",
#                 "message": str(e)
#             })
#         finally:
#             self._cleanup()
    
#     def _cleanup(self):
#         """Nettoyage avant arrêt"""
#         logger.info("="*70)
#         logger.info("ARRÊT DU SERVEUR D'INFÉRENCE")
#         logger.info("="*70)
        
#         stats = self.ai.get_stats()
#         logger.info(f"Statistiques finales:")
#         logger.info(f"  Prédictions totales: {stats['predictions_count']}")
#         logger.info(f"  Temps moyen: {stats['average_inference_time_ms']:.2f}ms")
#         logger.info(f"  Temps total: {stats['total_inference_time_s']:.2f}s")
#         logger.info(f"  Durée de vie: {time.time() - self.start_time:.2f}s")
        
#         logger.info("Serveur arrêté proprement")



# #                               POINT D'ENTRÉE


# def main():
#     """Point d'entrée principal du script"""
#     try:
#         # 1. Initialiser le pont IA
#         ai_bridge = AIBridge()
        
#         # 2. Créer le gestionnaire de communication
#         comm_handler = CommunicationHandler(ai_bridge)
        
#         # 3. Démarrer la boucle d'écoute
#         comm_handler.run()
        
#         # 4. Exit propre
#         sys.exit(0)
        
#     except FileNotFoundError as e:
#         logger.error(f"Fichier manquant: {e}")
#         print(json.dumps({
#             "status": "fatal_error",
#             "error_code": "MODEL_NOT_FOUND",
#             "message": str(e)
#         }), flush=True)
#         sys.exit(1)
        
#     except Exception as e:
#         logger.error(f"Erreur fatale au démarrage: {e}", exc_info=True)
#         print(json.dumps({
#             "status": "fatal_error",
#             "error_code": "STARTUP_FAILED",
#             "message": str(e)
#         }), flush=True)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
SMART-CONNECT4 - MODULE D'INFÉRENCE IA
Script d'interface entre l'agent DQN et le backend Node.js
Communication : StdIO/JSON
Protocole : Request/Response avec gestion d'erreurs robuste

Auteur : Mohamed NAJID
Projet : Smart-Connect4 - M2 IA UCBL

VERSION CORRIGÉE : Ajout détection coups gagnants/blocages
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Import des modules du projet
try:
    from src.dqn_agent import DQNAgent
    from src.dqn_model import DuelingDQN
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from dqn_agent import DQNAgent
    from dqn_model import DuelingDQN

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & LOGGING

class Config:
    """Configuration du module d'inférence"""
    # Chemins
    MODEL_PATH = "models/dqn_connect_four_final.pth"
    LOG_FILE = "logs/inference.log"
    
    # Paramètres du jeu
    INPUT_SHAPE = (3, 6, 7)
    NUM_ACTIONS = 7
    ROWS = 6
    COLS = 7
    
    # Optimisation
    USE_CPU = True
    TIMEOUT_PREDICTION = 5.0  # secondes
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_TO_FILE = False

def setup_logging():
    """Configure le système de logging"""
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler(sys.stderr)]
    
    if Config.LOG_TO_FILE:
        Path(Config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Config.LOG_FILE))
    
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ═══════════════════════════════════════════════════════════════════════════
# GESTION DU PLATEAU

class BoardManager:
    """Gestionnaire du plateau de jeu avec validation"""
    
    @staticmethod
    def validate_board(board):
        """
        Valide la structure et le contenu du plateau
        Args:
            board: Liste 2D représentant le plateau
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(board, list):
            return False, "Board must be a list"
        
        if len(board) != Config.ROWS:
            return False, f"Board must have {Config.ROWS} rows, got {len(board)}"
        
        for i, row in enumerate(board):
            if not isinstance(row, list):
                return False, f"Row {i} must be a list"
            
            if len(row) != Config.COLS:
                return False, f"Row {i} must have {Config.COLS} columns, got {len(row)}"
            
            for j, cell in enumerate(row):
                if cell not in [0, 1, 2]:
                    return False, f"Invalid cell value at ({i},{j}): {cell}. Must be 0, 1, or 2"
        
        return True, None
    
    @staticmethod
    def board_to_state(board):
        """
        Convertit le plateau JSON en tenseur (3, 6, 7) pour le réseau
        Args:
            board: Liste 2D (6x7) avec valeurs 0 (vide), 1 (joueur 1), 2 (joueur 2)
        Returns:
            torch.Tensor: Tenseur (1, 3, 6, 7) prêt pour l'inférence
        """
        board_array = np.array(board, dtype=np.float32)
        
        # Créer les 3 canaux
        state = np.zeros((3, Config.ROWS, Config.COLS), dtype=np.float32)
        state[0] = (board_array == 1).astype(np.float32)  # Joueur 1
        state[1] = (board_array == 2).astype(np.float32)  # Joueur 2
        state[2] = (board_array == 0).astype(np.float32)  # Cases vides
        
        # Ajouter dimension batch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return state_tensor
    
    @staticmethod
    def get_valid_actions(board):
        """
        Retourne les colonnes jouables (non pleines)
        Args:
            board: Liste 2D représentant le plateau
        Returns:
            list: Indices des colonnes jouables (0-6)
        """
        return [col for col in range(Config.COLS) if board[0][col] == 0]
    
    @staticmethod
    def get_board_stats(board):
        """Statistiques du plateau pour monitoring"""
        board_array = np.array(board)
        return {
            'player1_pieces': int(np.sum(board_array == 1)),
            'player2_pieces': int(np.sum(board_array == 2)),
            'empty_cells': int(np.sum(board_array == 0)),
            'valid_actions': len(BoardManager.get_valid_actions(board))
        }
    
    # ⚠️ NOUVEAU : Détection heuristique des coups critiques
    @staticmethod
    def detect_winning_move(board, player):
        """
        Détecte si un joueur peut gagner en 1 coup
        Args:
            board: Plateau de jeu (liste 2D)
            player: 1 ou 2
        Returns:
            int ou None: Colonne du coup gagnant, ou None
        """
        for col in range(Config.COLS):
            # Vérifier si la colonne n'est pas pleine
            if board[0][col] != 0:
                continue
            
            # Simuler le coup
            test_board = [row[:] for row in board]  # Copie profonde
            
            # Trouver où le jeton va tomber
            placed_row = None
            for row in range(Config.ROWS - 1, -1, -1):
                if test_board[row][col] == 0:
                    test_board[row][col] = player
                    placed_row = row
                    break
            
            # Vérifier si ce coup crée une victoire
            if placed_row is not None and BoardManager.check_win_at(test_board, placed_row, col, player):
                return col
        
        return None
    
    @staticmethod
    def check_win_at(board, row, col, player):
        """
        Vérifie si le coup en (row, col) crée une victoire (4 alignés)
        Args:
            board: Plateau de jeu
            row: Ligne du dernier coup
            col: Colonne du dernier coup
            player: Joueur (1 ou 2)
        Returns:
            bool: True si victoire
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonale descendante droite
            (1, -1)   # Diagonale descendante gauche
        ]
        
        for dr, dc in directions:
            count = 1  # Le jeton qu'on vient de placer
            
            # Compter dans les deux directions
            for direction in [1, -1]:
                r, c = row + dr * direction, col + dc * direction
                while 0 <= r < Config.ROWS and 0 <= c < Config.COLS and board[r][c] == player:
                    count += 1
                    r += dr * direction
                    c += dc * direction
            
            if count >= 4:
                return True
        
        return False

# ═══════════════════════════════════════════════════════════════════════════
# PONT IA (AI BRIDGE)

class AIBridge:
    """
    Pont entre le système Node.js et l'agent DQN
    Gère le chargement du modèle et les prédictions
    """
    
    def __init__(self, model_path=None):
        """
        Initialise le pont IA
        Args:
            model_path: Chemin vers le modèle PyTorch (.pth)
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.device = self._setup_device()
        self.agent = None
        self.load_time = None
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
        logger.info("=" * 70)
        logger.info("INITIALISATION DU MODULE IA")
        logger.info("=" * 70)
        logger.info(f"Device sélectionné: {self.device}")
        logger.info(f"Modèle: {self.model_path}")
        
        self._load_model()
    
    def _setup_device(self):
        """Configure le device (CPU/GPU) optimal"""
        if Config.USE_CPU:
            device = torch.device("cpu")
            logger.info("Mode CPU forcé pour latence prévisible")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                logger.info(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        return device
    
    def _load_model(self):
        """Charge le modèle DQN entraîné"""
        start_time = time.time()
        
        try:
            # Vérifier que le fichier existe
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
            
            logger.info(f"Chargement du modèle depuis {self.model_path}...")
            
            # Créer l'agent
            self.agent = DQNAgent(
                input_shape=Config.INPUT_SHAPE,
                num_actions=Config.NUM_ACTIONS,
                device=self.device
            )
            
            # Charger les poids
            self.agent.load_model(self.model_path)
            
            # Mode évaluation
            self.agent.policy_net.eval()
            self.agent.epsilon = 0.0  # Pas d'exploration en production
            
            self.load_time = time.time() - start_time
            
            logger.info(f"✓ Modèle chargé avec succès en {self.load_time:.3f}s")
            logger.info(f"  Architecture: {self.agent.policy_net.__class__.__name__}")
            logger.info(f"  Paramètres: {sum(p.numel() for p in self.agent.policy_net.parameters()):,}")
            
        except Exception as e:
            logger.error(f"✗ Erreur lors du chargement du modèle: {e}")
            raise
    
    def predict(self, board):
        """
        Prédit le meilleur coup à jouer
        Args:
            board: Plateau de jeu (liste 2D)
        Returns:
            dict: {column, confidence, inference_time_ms, ...}
        Raises:
            ValueError: Si le plateau est invalide
        """
        start_time = time.time()
        
        try:
            # 1. Validation du plateau
            is_valid, error_msg = BoardManager.validate_board(board)
            if not is_valid:
                raise ValueError(f"Plateau invalide: {error_msg}")
            
            # 2. Obtenir les actions valides
            valid_actions = BoardManager.get_valid_actions(board)
            if not valid_actions:
                raise ValueError("Aucune action valide: plateau plein")
            
            # ✅ 3. DÉTECTION COUP GAGNANT (Priorité 1)
            winning_move = BoardManager.detect_winning_move(board, player=1)
            if winning_move is not None and winning_move in valid_actions:
                inference_time = (time.time() - start_time) * 1000
                self.prediction_count += 1
                self.total_inference_time += inference_time
                
                logger.info(f"🎯 COUP GAGNANT DÉTECTÉ : Colonne {winning_move}")
                logger.info(f"   Confiance: 100.0%")
                logger.info(f"   Temps: {inference_time:.2f}ms")
                
                return {
                    "column": winning_move,
                    "confidence": 1.0,
                    "q_values": [0.0] * Config.NUM_ACTIONS,
                    "valid_actions": valid_actions,
                    "inference_time_ms": round(inference_time, 2),
                    "board_stats": BoardManager.get_board_stats(board),
                    "strategy": "winning_move"
                }
            
            # ✅ 4. DÉTECTION BLOCAGE URGENT (Priorité 2)
            blocking_move = BoardManager.detect_winning_move(board, player=2)
            if blocking_move is not None and blocking_move in valid_actions:
                inference_time = (time.time() - start_time) * 1000
                self.prediction_count += 1
                self.total_inference_time += inference_time
                
                logger.info(f"🛡️ BLOCAGE URGENT : Colonne {blocking_move}")
                logger.info(f"   Confiance: 95.0%")
                logger.info(f"   Temps: {inference_time:.2f}ms")
                
                return {
                    "column": blocking_move,
                    "confidence": 0.95,
                    "q_values": [0.0] * Config.NUM_ACTIONS,
                    "valid_actions": valid_actions,
                    "inference_time_ms": round(inference_time, 2),
                    "board_stats": BoardManager.get_board_stats(board),
                    "strategy": "blocking_move"
                }
            
            # 5. RÉSEAU DE NEURONES (Si pas de coup critique)
            state_tensor = BoardManager.board_to_state(board).to(self.device)
            
            with torch.no_grad():
                # Obtenir les Q-values brutes
                q_values = self.agent.policy_net(state_tensor).squeeze(0)
                
                # Masquer les actions invalides
                mask = torch.full_like(q_values, -float('inf'))
                for action_idx in valid_actions:
                    mask[action_idx] = 0
                masked_q_values = q_values + mask
                
                # Sélectionner la meilleure action
                best_action = masked_q_values.argmax().item()
                confidence = torch.softmax(masked_q_values[valid_actions], dim=0).max().item()
            
            # 6. Métriques
            inference_time = (time.time() - start_time) * 1000
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            # 7. Logging
            stats = BoardManager.get_board_stats(board)
            logger.info(f"Prédiction #{self.prediction_count}")
            logger.info(f"  État: J1={stats['player1_pieces']} | J2={stats['player2_pieces']} | Vide={stats['empty_cells']}")
            logger.info(f"  Actions valides: {valid_actions}")
            logger.info(f"  Coup choisi: Colonne {best_action}")
            logger.info(f"  Confiance: {confidence:.1%}")
            logger.info(f"  Temps: {inference_time:.2f}ms")
            
            # 8. Retour
            return {
                "column": best_action,
                "confidence": round(confidence, 4),
                "q_values": [round(float(q), 3) for q in q_values.cpu().tolist()],
                "valid_actions": valid_actions,
                "inference_time_ms": round(inference_time, 2),
                "board_stats": stats,
                "strategy": "neural_network"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
            raise
    
    def get_stats(self):
        """Retourne les statistiques d'utilisation"""
        avg_time = self.total_inference_time / self.prediction_count if self.prediction_count > 0 else 0
        return {
            "predictions_count": self.prediction_count,
            "average_inference_time_ms": round(avg_time, 2),
            "total_inference_time_s": round(self.total_inference_time / 1000, 2),
            "model_load_time_s": round(self.load_time, 3) if self.load_time else None,
            "device": str(self.device)
        }

# ═══════════════════════════════════════════════════════════════════════════
# GESTIONNAIRE DE COMMUNICATION

class CommunicationHandler:
    """
    Gestionnaire de communication StdIO avec Node.js
    Protocole: JSON lines
    """
    
    def __init__(self, ai_bridge):
        self.ai = ai_bridge
        self.request_count = 0
        self.start_time = time.time()
        logger.info("Gestionnaire de communication initialisé")
    
    def send_response(self, response):
        """Envoie une réponse JSON sur stdout"""
        try:
            json_response = json.dumps(response)
            print(json_response, flush=True)
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la réponse: {e}")
            print(json.dumps({"status": "error", "message": "Response serialization failed"}), flush=True)
    
    def handle_predict(self, request):
        """Gère une requête de prédiction"""
        try:
            board = request.get("board")
            if board is None:
                return {
                    "status": "error",
                    "error_code": "MISSING_BOARD",
                    "message": "Missing 'board' field in request"
                }
            
            result = self.ai.predict(board)
            
            return {
                "status": "success",
                "column": result["column"],
                "metadata": {
                    "confidence": result["confidence"],
                    "inference_time_ms": result["inference_time_ms"],
                    "valid_actions": result["valid_actions"],
                    "board_stats": result["board_stats"],
                    "strategy": result.get("strategy", "neural_network"),
                    "timestamp": datetime.now().isoformat()
                },
                "debug": {
                    "q_values": result["q_values"]
                } if logger.level == logging.DEBUG else {}
            }
            
        except ValueError as e:
            return {
                "status": "error",
                "error_code": "INVALID_BOARD",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Erreur inattendue dans handle_predict: {e}", exc_info=True)
            return {
                "status": "error",
                "error_code": "PREDICTION_FAILED",
                "message": f"Unexpected error: {str(e)}"
            }
    
    def handle_ping(self, request):
        """Health check"""
        uptime = time.time() - self.start_time
        return {
            "status": "pong",
            "uptime_seconds": round(uptime, 2),
            "stats": self.ai.get_stats()
        }
    
    def handle_stats(self, request):
        """Retourne les statistiques"""
        return {
            "status": "success",
            "stats": self.ai.get_stats(),
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "total_requests": self.request_count
        }
    
    def handle_request(self, line):
        """Traite une requête JSON"""
        self.request_count += 1
        
        try:
            request = json.loads(line.strip())
            command = request.get("command", "predict")
            
            if command == "predict":
                return self.handle_predict(request)
            elif command == "ping":
                return self.handle_ping(request)
            elif command == "stats":
                return self.handle_stats(request)
            elif command == "shutdown":
                logger.info("Commande de shutdown reçue")
                return {"status": "shutdown", "message": "Shutting down gracefully"}
            else:
                return {
                    "status": "error",
                    "error_code": "UNKNOWN_COMMAND",
                    "message": f"Unknown command: {command}"
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            return {
                "status": "error",
                "error_code": "INVALID_JSON",
                "message": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}", exc_info=True)
            return {
                "status": "error",
                "error_code": "INTERNAL_ERROR",
                "message": f"Internal error: {str(e)}"
            }
    
    def run(self):
        """Boucle principale d'écoute StdIO"""
        logger.info("=" * 70)
        logger.info("SERVEUR D'INFÉRENCE DÉMARRÉ")
        logger.info("=" * 70)
        logger.info("En attente de requêtes sur stdin...")
        
        # Signal de readiness
        self.send_response({
            "status": "ready",
            "message": "AI inference server ready",
            "config": {
                "model": Config.MODEL_PATH,
                "device": str(self.ai.device),
                "version": "1.0.1"  # ✅ Version mise à jour
            }
        })
        
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                
                response = self.handle_request(line)
                self.send_response(response)
                
                if response.get("status") == "shutdown":
                    break
            
            logger.info("Sortie de la boucle d'écoute (stdin fermé)")
            
        except KeyboardInterrupt:
            logger.info("Interruption clavier détectée (Ctrl+C)")
        except Exception as e:
            logger.error(f"Erreur fatale dans la boucle principale: {e}", exc_info=True)
            self.send_response({
                "status": "fatal_error",
                "message": str(e)
            })
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Nettoyage avant arrêt"""
        logger.info("=" * 70)
        logger.info("ARRÊT DU SERVEUR D'INFÉRENCE")
        logger.info("=" * 70)
        stats = self.ai.get_stats()
        logger.info(f"Statistiques finales:")
        logger.info(f"  Prédictions totales: {stats['predictions_count']}")
        logger.info(f"  Temps moyen: {stats['average_inference_time_ms']:.2f}ms")
        logger.info(f"  Temps total: {stats['total_inference_time_s']:.2f}s")
        logger.info(f"  Durée de vie: {time.time() - self.start_time:.2f}s")
        logger.info("Serveur arrêté proprement")

# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE

def main():
    """Point d'entrée principal"""
    try:
        ai_bridge = AIBridge()
        comm_handler = CommunicationHandler(ai_bridge)
        comm_handler.run()
        sys.exit(0)
        
    except FileNotFoundError as e:
        logger.error(f"Fichier manquant: {e}")
        print(json.dumps({
            "status": "fatal_error",
            "error_code": "MODEL_NOT_FOUND",
            "message": str(e)
        }), flush=True)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Erreur fatale au démarrage: {e}", exc_info=True)
        print(json.dumps({
            "status": "fatal_error",
            "error_code": "STARTUP_FAILED",
            "message": str(e)
        }), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
