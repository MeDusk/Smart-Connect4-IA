# Smart-Connect4-IA

## Structure du Projet

```
Smart-Connect4-IA/
├── src/
│   ├── __init__.py
│   ├── dqn_agent.py              # Agent DQN avec Dueling architecture
│   ├── dqn_model.py              # Architecture Dueling DQN
│   ├── connect_four_env.py       # Environnement de simulation Puissance 4
│   ├── train_dqn.py              # Script d'entraînement principal
│   └── test_dqn.py               # Tests de l'agent
├── inference.py                   # Module d'inférence pour production
├── test_inference.py              # Tests d'intégration backend
├── requirements.txt               # Dépendances Python
├── models/
│   └── dqn_connect_four_final.pth # Modèle entraîné (Git LFS)
└── README.md                      # Documentation
```

***

## Commandes d'Utilisation

### 1️Installation

```bash
# Cloner le dépôt GitHub
git clone https://github.com/MeDusk/Smart-Connect4-IA.git
cd Smart-Connect4-IA

# Créer un environnement virtuel Python
python -m venv .venv

# Activer l'environnement virtuel
.venv\Scripts\activate             

# Installer les dépendances
pip install -r requirements.txt
```


### Inférence 

#### Lancer le Serveur d'Inférence

```bash
# Démarrer le module IA en mode production
python inference.py
```

**Sortie attendue :**
```json
{"status": "ready", "message": "AI inference server ready", "config": {...}}
```

Ca veut dire que Le serveur attend les requêtes sur **stdin** (entrée standard).

***

## Tester l'Inférence Manuellement

### Commande de Base

```bash
echo '{"command":"predict","board":[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]}' | python inference.py
```

***

### 📖 Explication de la Commande

#### Structure de la Requête JSON

```json
{
  "command": "predict",
  "board": [
    [0, 0, 0, 0, 0, 0, 0],  ← Ligne 0 (en haut)
    [0, 0, 0, 0, 0, 0, 0],  ← Ligne 1
    [0, 0, 0, 0, 0, 0, 0],  ← Ligne 2
    [0, 0, 0, 0, 0, 0, 0],  ← Ligne 3
    [0, 0, 0, 0, 0, 0, 0],  ← Ligne 4
    [0, 0, 0, 0, 0, 0, 0]   ← Ligne 5 (en bas)
  ]
}
```

#### Signification des Valeurs

| Valeur | Signification |
|--------|--------------|
| `0` | Case vide |
| `1` | Jeton du IA |
| `2` | Jeton de Humain |

***

### Exemples de Plateaux

#### Exemple 1 : Plateau Vide (Début de Partie)

```bash
echo '{"command":"predict","board":[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]}' | python inference.py
```

**Plateau visualisé :**
```
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
```

**Réponse attendue :**
```json
{
  "status": "success",
  "column": 3,
  "metadata": {
    "confidence": 0.92,
    "inference_time_ms": 4.2,
    "valid_actions": [0, 1, 2, 3, 4, 5, 6]
  }
}
```
→ L'IA choisit la colonne 3 (centre du plateau)

***

#### Exemple 2 : Partie en Cours

```bash
echo '{"command":"predict","board":[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,2,1,0,0,0]]}' | python inference.py
```

**Plateau visualisé :**
```
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | . | . | . | . | . |
| . | . | X | . | . | . | . |  ← Ligne 4 : 1 jeton Humain (X)
| . | . | O | X | . | . | . |  ← Ligne 5 : 1 jeton IA (O), 1 jeton Humain (X)
  0   1   2   3   4   5   6    ← Numéros de colonnes
```

**Légende :**
- `X` = Joueur 1 (IA) = `1` dans le JSON
- `O` = Joueur 2 (Humain) = `2` dans le JSON
- `.` = Case vide = `0` dans le JSON

**Réponse attendue :**
```json
{
  "status": "success",
  "column": 2,
  "metadata": {
    "confidence": 0.87,
    "inference_time_ms": 4.1,
    "valid_actions": [0, 1, 2, 3, 4, 5, 6]
  }
}
```
→ L'IA joue en colonne 2 pour construire une menace verticale

***



### Commande Shutdown (Arrêt Propre)

```bash
echo '{"command":"shutdown"}' | python inference.py
```

**Réponse :**
```json
{
  "status": "shutdown",
  "message": "AI server shutting down gracefully"
}
```
