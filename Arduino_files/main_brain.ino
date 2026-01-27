/*
 * Projet : Robot Puissance 4 - AI for IoT and Robotics
 * Plateforme : Arduino Mega
 * Description : Contrôle bas niveau (Capteurs + Moteurs) et interface Série avec Python
 */
#include <Servo.h>
#include <Stepper.h>

// --- CONFIGURATION STEPPER MOTOR 28BYJ-48 ---
const int PAS_PAR_TOUR = 2038;
const int DISTANCE_COLONNE = 1111; // À ajuster selon La grille
const int VITESSE_RPM = 10; // Vitesse en tours/minute (Max 15 pour ce moteur)

// --- CONFIGURATION SERVO MOTOR ---
const int COIN_INIT_POS = 70; // Position initiale du dispenseur de pièces
const int COIN_RELEASE_POS = 0; // Position de larguage du dispenseur de pièces

// --- PINS ---
#define STEPPER_PIN_1 1 // IN4 sur le driver ULN2003
#define STEPPER_PIN_2 2 // IN3
#define STEPPER_PIN_3 3 // IN2
#define STEPPER_PIN_4 4 // IN1
#define SERVO_PIN 6     // Fil signal du servo

const int PINS_SENSORS[7] = {A6, A5, A4, A3, A2, A1, A0}; // Exemple à adapter selon le câblage

Stepper monMoteur(PAS_PAR_TOUR, STEPPER_PIN_1, STEPPER_PIN_2, STEPPER_PIN_3, STEPPER_PIN_4);
Servo monServo;

// --- PARAMÈTRES ---
const float FACTEUR_DETECTION = 0.4; // Seuil de luminosité 
int calibrations[7];             // Valeurs de référence (lumière ambiante)

// --- ÉTATS DU SYSTÈME ---
enum State {
  INITIALISATION,
  TOUR_JOUEUR,
  ENVOI_COUP_JOUEUR,
  REFLEXION_IA,
  TOUR_ROBOT,
  FIN_PARTIE
};

State etatActuel = INITIALISATION;
int colonneJoueeParHumain = -1;
int colonneAJouerParRobot =  -1;

void setup() {
  Serial.begin(9600); // Vitesse à aligner avec le script Python

  // Config des moteurs
  monServo.attach(SERVO_PIN);
  monServo.write(COIN_INIT_POS); // Fermé
  
  monMoteur.setSpeed(VITESSE_RPM); // Règle la vitesse
  
  // Phase de calibration : on lit la lumière ambiante au démarrage
  Serial.println("Calibration des capteurs...");
  calibrerCapteurs();
  Serial.println("Test des moteurs...");
  testerMoteurs();
  Serial.println("Pret. En attente du joueur.");
  etatActuel = TOUR_JOUEUR;
}

void loop() {
  switch (etatActuel) {
    
    // --- PHASE 1 : C'EST À L'HUMAIN DE JOUER ---
    case TOUR_JOUEUR:
      colonneJoueeParHumain = scannerColonnes();
      if (colonneJoueeParHumain != -1) {
        // Anti-rebond basique (attendre que le pion finisse de tomber)
        delay(100); 
        etatActuel = ENVOI_COUP_JOUEUR;
        
      }
      break;

    // --- PHASE 2 : TRANSMISSION À PYTHON ---
    case ENVOI_COUP_JOUEUR:
      // Protocole simple : "P" + numéro colonne (ex: "P3")
      Serial.print("P");
      Serial.println(colonneJoueeParHumain);
      etatActuel = REFLEXION_IA;
      break;

    // --- PHASE 3 : ATTENTE DE L'IA ---
    case REFLEXION_IA:
      if (Serial.available() > 0) {
        char cmd = Serial.read();
        
        // CAS 1 : LE ROBOT JOUE (Jeu continue)
        if (cmd == 'R') {
          colonneAJouerParRobot = Serial.parseInt();
          etatActuel = TOUR_ROBOT;
        }
        
        // CAS 2 : VICTOIRE DU ROBOT ("V" + Colonne)
        else if (cmd == 'V') {
          colonneAJouerParRobot = Serial.parseInt();
          deplacerEtLacherPion(colonneAJouerParRobot); // Joue le coup gagnant
          celebrerVictoire();
          etatActuel = FIN_PARTIE;
        }
        
        // CAS 3 : DÉFAITE DU ROBOT / VICTOIRE JOUEUR ("D")
        else if (cmd == 'D') {
          // Le joueur vient de jouer, pas de mouvement robot à faire
          signalerDefaite();
          etatActuel = FIN_PARTIE;
        }
        
        // CAS 4 : MATCH NUL ("N" ou "N"+Colonne)
        else if (cmd == 'N') {
          // Subtilité : Nul causé par Joueur ("N") ou par Robot ("N3") ?
          delay(50); // Petite pause pour laisser le temps au buffer de recevoir le chiffre
          
          if (Serial.available() > 0 && isDigit(Serial.peek())) {
            // C'est un chiffre qui suit -> Le robot joue le dernier pion
            colonneAJouerParRobot = Serial.parseInt();
            deplacerEtLacherPion(colonneAJouerParRobot);
          }
          // Sinon (pas de chiffre), c'est le joueur qui a rempli la grille. Rien à faire.
          signalerNul();
          etatActuel = FIN_PARTIE;
        }
      }
      break;

    // --- PHASE 4 : MOUVEMENT MÉCANIQUE ---
    case TOUR_ROBOT:
      deplacerEtLacherPion(colonneAJouerParRobot);
      // Une fois fini, on informe Python (optionnel) ou on repasse la main
      Serial.println("DONE"); // Ack pour dire que le mouvement est fini
      Serial.println("En attente du joueur.");
      etatActuel = TOUR_JOUEUR;
      break;

    // --- PHASE 5 : FIN DE PARTIE ---
    case FIN_PARTIE:
      // On bloque ici jusqu'à un reset (manuel ou via Serial plus tard)
      attendreReset();
      break;
  }
}

// --- FONCTIONS UTILITAIRES ---

void testerMoteurs() {
  monMoteur.step(DISTANCE_COLONNE);
  delay(100);
  monMoteur.step(-DISTANCE_COLONNE);
  delay(100);
  monServo.write(0);
  delay(100);
}

void calibrerCapteurs() {
  for (int i = 0; i < 7; i++) {
    // Lecture moyenne sur 10 échantillons pour la stabilité
    long somme = 0;
    for(int j=0; j<20; j++) {
      somme += analogRead(PINS_SENSORS[i]);
      delay(10);
    }
    calibrations[i] = somme / 20;
    Serial.println(somme / 20);
  }
}

int scannerColonnes() {
  // Scanne les 7 capteurs pour voir si une valeur chute (passage d'ombre)
  for (int i = 0; i < 7; i++) {
    int valeurActuelle = analogRead(PINS_SENSORS[i]);
    // Si la valeur chute brutalement par rapport à la calibration (pion passe devant)
    // Note : adapter le sens (< ou >) selon le type de photorésistance/montage
    if (valeurActuelle < (calibrations[i] * FACTEUR_DETECTION)) {
      return i; // Retourne l'index de la colonne (0 à 6)
    }
  }
  return -1; // Rien détecté
}

void deplacerEtLacherPion(int colonne) {
  Serial.print("Deplacement vers colonne ");
  Serial.println(colonne);
  monMoteur.step((colonne+1)*DISTANCE_COLONNE);
  lacherPiece();
  monMoteur.step(-(colonne+1)*DISTANCE_COLONNE);
}

void lacherPiece() {
  Serial.println("Largage !");
  for (int i = 0; i < 10; i++) {
  monServo.write((i+1)*COIN_RELEASE_POS/10);
  }
  delay(1000);
  monServo.write(COIN_INIT_POS);
  delay(500);
}

void celebrerVictoire() {
  Serial.println("--- VICTOIRE ROBOT ! ---");
  // Idée : Faire des aller-retours rapides avec le chariot ou bouger le servo
  for(int i=0; i<3; i++) {
    // Danse de la joie
    Serial.println("Danse...");
    delay(200);
  }
}

void signalerDefaite() {
  Serial.println("--- DEFAITE ROBOT ---");
  // Idée : Mouvement lent ou bip grave
}

void signalerNul() {
  Serial.println("--- MATCH NUL ---");
}

void attendreReset() {
  // Boucle infinie. Appuyer sur le bouton Reset de l'Arduino pour relancer.
  // Ou attendre une commande 'RESET' de Python si tu veux automatiser.
  while(true) {
    delay(1000);
  }
}
