# Robot Puissance 4 Autonome

| | |
|---|---|
| **Élèves** | [Vos noms] |
| **Professeur** | [Nom du prof] |
| **Unité d'Enseignement** | [Code UE - Titre] |
| **Établissement** | Université Claude Bernard Lyon 1 |
| **Période** | [Date de début - Date de fin] |

---
## Introduction

Ce projet vise à développer un **robot autonome capable de jouer au Puissance 4** contre un adversaire humain. Le système combine trois éléments clés :
- Une **intelligence artificielle** entraînée par apprentissage par renforcement (DQN) capable de jouer au puissance 4
- Un **contrôleur Arduino** gérant les capteurs et actionneurs physiques
- Une **interface web** permettant de visualiser et contrôler le jeu en temps réel

[**Insérer image du montage physique ici**]

---
## Structure du Projet

```
Smart-Connect4-IA/
│
├── ai_agent/         # IA de jeu
│   ├── brain.py      # API Flask de l'IA
│   ├── src/          # Scripts d'entrainement et de test des modèles
│   ├── models/       # Poids des modèles entraînés
│   └── requirements.txt
│
├── arduino_controler/ # Contrôle du hardware
│   ├── main_brain.ino # Code initial Arduino pour tester les actionneurs/capteurs
│   └── firmata.js     # Pont communiquant entre la arduino (via firmata), l'API du programme de jeu et le frontend (WebSocket)
│
├── web_client/       # Frontend
│   ├── src/          # Scripts React
│   ├── index.html    # Page HTML
│   ├── package.json  # Dépendances Node.js
│   └── vite.config.js
│
└── README.md
```

---
## Composants du Projet

### 1 - Module IA (`ai_agent/`)

Implémentation d'un agent DQN Dueling pour apprendre la stratégie optimale au Puissance 4.

**Responsabilités :**
- `connect_four_env.py` : Implémentation de l'environnement de jeu (règles, états, récompenses)
- `train_dqn.py` : Entraînement de l'agent contre lui-même via apprentissage par renforcement
- `inference.py` : Module d'inférence qui charge le modèle et effectue les prédictions
- `brain.py` : Serveur Flask exposant l'endpoint `/predict` pour les requêtes HTTP

**Technologies :** PyTorch, NumPy, Flask

---

### 2 - Contrôleur Arduino (`arduino_controler/`)

Gestion du hardware physique : capteurs de détection de pièces et moteurs de positionnement.

**Responsabilités :**
- `main_brain.ino` : Code Arduino bas niveau pour :
  - Lecture des capteurs infrarouge (7 colonnes)
  - Contrôle du moteur pas à pas (mouvement horizontal)
  - Contrôle du servo moteur (libération des pièces)

- `firmata.js` : adaptation de la logique de jeu implémentée précédemment qui assurant le pont entre :
  - Arduino (via Firmata protocol)
  - Serveur Python IA (appels HTTP)
  - Interface web (WebSocket)

**Technologies :** Arduino C++, Johnny-Five (Node.js), Firmata Protocol

---

### 3 - Interface Web (`web_client/`)

Application React affichant l'état du plateau et communiquant avec les services backend.

**Responsabilités :**
- `App.jsx` : Composant principal affichant :
  - Le plateau de jeu 6x7 (avec animations Framer Motion)
  - L'état de la partie en temps réel
  - Bouton de réinitialisation
- Communication WebSocket avec le serveur Arduino Bridge
- Vite : bundler rapide pour le développement et la production

**Technologies :** React 19, Framer Motion, Vite, WebSocket

---

## Guide de Démarrage
