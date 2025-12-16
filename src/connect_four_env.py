import numpy as np

class ConnectFourEnv:
    """
    Environnement de simulation pour le jeu de Puissance 4.
    Modélise l'état du plateau, les actions, les récompenses et les règles du jeu.
    """
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # 1 pour l'IA (joueur principal), 2 pour l'adversaire
        self.game_over = False
        self.winner = 0

    def reset(self):
        """ Réinitialise le plateau et l'état du jeu. """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        return self._get_state()

    def _get_state(self):
        """
        Représentation de l'état du plateau pour le réseau neuronal.
        Utilise une représentation 3D (3 canaux) :
        - Canal 0 : Position des jetons du joueur 1 (IA)
        - Canal 1 : Position des jetons du joueur 2 (Adversaire)
        - Canal 2 : Cellules vides
        """
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.board == 1)
        state[1] = (self.board == 2)
        state[2] = (self.board == 0)
        return state

    def get_valid_actions(self):
        """ Retourne la liste des colonnes jouables (non pleines). """
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def drop_piece(self, col, player):
        """ Dépose un jeton dans la colonne spécifiée pour le joueur donné. """
        if col not in self.get_valid_actions():
            raise ValueError(f"Colonne {col} invalide ou pleine.")
        
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return row, col
        return -1, col # Ne devrait pas arriver si la colonne est valide

    def check_win(self, player):
        """ Vérifie si le joueur a gagné. """
        # Horizontal, Vertical, Diagonales (montante et descendante)
        
        # Helper function for checking a direction
        def check_direction(r_start, c_start, r_delta, c_delta):
            count = 0
            for i in range(self.win_length):
                r, c = r_start + i * r_delta, c_start + i * c_delta
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    count = 0
                if count == self.win_length:
                    return True
            return False

        # Check all possible starting positions for a win_length line
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == player:
                    # Check horizontal (to the right)
                    if c <= self.cols - self.win_length:
                        if all(self.board[r, c+i] == player for i in range(self.win_length)):
                            return True
                    # Check vertical (down)
                    if r <= self.rows - self.win_length:
                        if all(self.board[r+i, c] == player for i in range(self.win_length)):
                            return True
                    # Check diagonal (down-right)
                    if r <= self.rows - self.win_length and c <= self.cols - self.win_length:
                        if all(self.board[r+i, c+i] == player for i in range(self.win_length)):
                            return True
                    # Check diagonal (down-left)
                    if r <= self.rows - self.win_length and c >= self.win_length - 1:
                        if all(self.board[r+i, c-i] == player for i in range(self.win_length)):
                            return True
        return False

    def check_draw(self):
        """ Vérifie si le plateau est plein (match nul). """
        return not self.get_valid_actions() and not self.winner

    def step(self, action):
        """Exécute une action avec récompenses améliorées"""
        if self.game_over:
            return self._get_state(), 0.0, True, {}
    
        if action not in self.get_valid_actions():
            self.game_over = True
            return self._get_state(), -100.0, True, {"invalid": True}
    
    # Exécuter l'action
        row, col = self.drop_piece(action, self.current_player)
    
    # Vérifier victoire
        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 100.0 if self.current_player == 1 else -100.0
            return self._get_state(), reward, True, {"winner": self.current_player}
    
    # Vérifier match nul
        if self.check_draw():
            self.game_over = True
            return self._get_state(), 0.0, True, {"draw": True}
    
    # NOUVEAU : Récompenses intermédiaires intelligentes
        reward = self._calculate_strategic_reward(row, col, self.current_player)
    
    # Changer de joueur
        self.current_player = 3 - self.current_player
    
        return self._get_state(), reward, False, {}

    def render(self):
        """ Affiche le plateau de jeu. """
        print("\n" + "=" * (self.cols * 4 + 1))
        print(" " + " | ".join(map(str, range(self.cols))))
        print("=" * (self.cols * 4 + 1))
        
        # Affichage des jetons: 1 -> X (IA), 2 -> O (Adversaire), 0 -> . (Vide)
        display_board = np.where(self.board == 1, 'X', np.where(self.board == 2, 'O', '.'))
        
        for row in display_board:
            print("| " + " | ".join(row) + " |")
            print("-" * (self.cols * 4 + 1))
        
        if self.game_over:
            if self.winner == 1:
                print("VICTOIRE de l'IA (X)!")
            elif self.winner == 2:
                print("VICTOIRE de l'Adversaire (O)!")
            else:
                print("MATCH NUL!")
        print("=" * (self.cols * 4 + 1) + "\n")
        
    def _calculate_strategic_reward(self, row, col, player):
        """Calcule récompenses basées sur la stratégie"""
        reward = 0.0
    
    # Vérifie les alignements pour ce joueur
        reward += self._count_threats(row, col, player) * 10.0
    
    # Pénalise si on n'a pas bloqué une menace adverse
        opponent = 3 - player
        opponent_threats = self._count_all_threats(opponent)
        if opponent_threats > 0:
            reward -= opponent_threats * 5.0
    
    # Récompense position centrale
        center_col = self.cols // 2
        if abs(col - center_col) <= 1:
            reward += 2.0
    
        return reward

    def _count_threats(self, row, col, player):
        """Compte le nombre d'alignements de 3 créés"""
        count = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
    
        for dr, dc in directions:
            line_count = 1
        # Direction positive
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r,c] == player:
                line_count += 1
                r += dr
                c += dc
        # Direction négative
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r,c] == player:
                line_count += 1
                r -= dr
                c -= dc
        
            if line_count == 3:
                count += 1
    
        return count

    def _count_all_threats(self, player):
        """Compte toutes les menaces de victoire pour un joueur"""
        threats = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == player:
                    threats += self._count_threats(r, c, player)
        return threats       

# Exemple d'utilisation 
if __name__ == '__main__':
    env = ConnectFourEnv()
    state = env.reset()
    env.render()
    
    # Simuler quelques coups
    # Coup 1 (IA)
    state, reward, done, info = env.step(3)
    env.render()
    
    # Coup 2 (Adversaire - doit être géré manuellement ou par un agent adversaire)
    # Pour simuler un tour complet (IA -> Adversaire -> IA)
    
    # Pour le test unitaire de l'environnement, on peut forcer le joueur 2 à jouer:
    env.current_player = 2
    state, reward, done, info = env.step(3) # Joue dans la même colonne
    env.render()
    
    # Retour à l'IA
    env.current_player = 1
    state, reward, done, info = env.step(4)
    env.render()
    
    # Forcer une victoire de l'IA (colonne 0)
    env.reset()
    env.drop_piece(0, 1)
    env.drop_piece(1, 2)
    env.drop_piece(0, 1)
    env.drop_piece(1, 2)
    env.drop_piece(0, 1)
    env.drop_piece(1, 2)
    
    # Coup gagnant de l'IA
    state, reward, done, info = env.step(0)
    env.render()
    print(f"Récompense: {reward}, Terminé: {done}")
    
    # Test du match nul
    env.reset()
    # Remplir le plateau sans victoire 
    
    # Test des actions invalides
    env.reset()
    for _ in range(6):
        env.drop_piece(0, 1)
    try:
        env.step(0)
    except ValueError as e:
        print(f"Erreur attendue pour colonne pleine: {e}")
