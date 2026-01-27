import { WebSocketServer } from 'ws';
import axios from 'axios';
import pkg from 'johnny-five';
const { Board, Sensor } = pkg;

// --- CONFIGURATION ---
const PORT_NODE = 8085;
const URL_PYTHON = 'http://127.0.0.1:5000/predict';

// RÉGLAGES CAPTEURS
const CALIBRATION_TIME = 2000;
const SENSOR_FREQ = 20;
const DETECTION_THRESHOLD = 350;

// --- VARIABLES GLOBALES ---
const ROWS = 6;
const COLS = 7;
const HUMAN = 1;
const AI = -1;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let isAiThinking = false;
let isCalibrating = true;

// Stockage pour la calibration
let sensorsBaselines = [0, 0, 0, 0, 0, 0, 0]; // La "moyenne" de lumière par trou
let calibrationSamples = [[], [], [], [], [], [], []]; // Echantillons temporaires
let triggers = [false, false, false, false, false, false, false]; // État du capteur (bloqué ou pas)

// --- WEBSOCKET ---
const wss = new WebSocketServer({ port: PORT_NODE });

function broadcast(data) {
    wss.clients.forEach(client => {
        if (client.readyState === 1) client.send(JSON.stringify(data));
    });
}

// --- LOGIQUE JEU ---
function playMove(col, player) {
    if (board[0][col] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r][col] === 0) {
            board[r][col] = player;
            return true;
        }
    }
    return false;
}

function resetGame() {
    board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
    isAiThinking = false;
    broadcast({ action: "RESET_DONE" });
    console.log("🔄 Plateau réinitialisé.");
}

// --- ARDUINO & JOHNNY-FIVE ---
const arduino = new Board({ repl: false });

arduino.on("ready", () => {
    console.log("✅ Arduino Connecté !");
    console.log("⏳ DÉBUT CALIBRATION (Ne rien toucher pendant 2 secondes)...");

    const sensors = [];

    // Initialisation des 7 capteurs
    for (let i = 0; i < 7; i++) {
        // A0 à A6
        sensors[i] = new Sensor({ pin: `A${i}`, freq: SENSOR_FREQ });

        sensors[i].on("data", async function() {
            const val = this.value;

            // --- PHASE 1 : CALIBRATION ---
            if (isCalibrating) {
                calibrationSamples[i].push(val);
                return;
            }

            // --- PHASE 2 : JEU ---
            // On compare la valeur actuelle à la moyenne calibrée
            // Note : En général, quand on cache la lumière, la valeur change.
            // On utilise Math.abs pour détecter un changement brusque (haut ou bas selon ton circuit)
            const diff = Math.abs(val - sensorsBaselines[i]);

            // DÉTECTION PASSAGE JETON
            if (diff > DETECTION_THRESHOLD) {
                if (!triggers[i]) {
                    // C'est le front montant (le jeton commence à passer)
                    triggers[i] = true; 
                    console.log(`🔻 PASSAGE DÉTECTÉ COLONNE ${i} (Val: ${val} | Base: ${sensorsBaselines[i]})`);
                    
                    if (isAiThinking) {
                        console.log("⚠️ Ignoré : L'IA réfléchit encore.");
                        return;
                    }

                    // 1. Jouer le coup Humain
                    if (playMove(i, HUMAN)) {
                        broadcast({ couleur: "Jaune", colonne: i.toString() });
                        isAiThinking = true;

                        // 2. Demander à l'IA
                        try {
                            // Petit délai pour laisser le jeton physique finir de tomber
                            setTimeout(async () => {
                                const response = await axios.post(URL_PYTHON, { board: board });
                                const aiCol = response.data.column;
                                
                                console.log(`🤖 IA joue Colonne ${aiCol}`);

                                // Simuler le temps de mouvement du robot
                                setTimeout(() => {
                                    if (playMove(aiCol, AI)) {
                                        broadcast({ couleur: "Rouge", colonne: aiCol.toString() });
                                        
                                        // ICI : CODE SERVOS POUR LACHER LE JETON IA
                                        // moveServo(aiCol);

                                        isAiThinking = false;
                                        console.log("✅ Tour terminé.");
                                    }
                                }, 1000);
                            }, 500);

                        } catch (e) {
                            console.error("❌ Erreur IA:", e.message);
                            isAiThinking = false;
                        }
                    } else {
                        console.log("⚠️ Colonne pleine !");
                    }
                }
            } 
            else {
                // Le capteur est revenu à la normale (le jeton est passé)
                // On ajoute une petite marge (hysteresis) pour éviter les double-clics
                if (diff < (DETECTION_THRESHOLD / 2)) {
                    triggers[i] = false;
                }
            }
        });
    }

    // Fin de la calibration après X secondes
    setTimeout(() => {
        console.log("📊 Fin Calibration. Calcul des moyennes...");
        for (let i = 0; i < 7; i++) {
            const samples = calibrationSamples[i];
            if (samples.length > 0) {
                const sum = samples.reduce((a, b) => a + b, 0);
                sensorsBaselines[i] = Math.floor(sum / samples.length);
                console.log(`   Col ${i} : Base = ${sensorsBaselines[i]}`);
            }
        }
        isCalibrating = false;
        console.log("🟢 JEU PRÊT ! À vous de jouer.");
    }, CALIBRATION_TIME);
});


// --- INTERFACE WEB ---
wss.on('connection', ws => {
    console.log("💻 Interface Web connectée");
    ws.on('message', msg => {
        const d = JSON.parse(msg);
        if (d.action === "RESET") resetGame();
    });
});

console.log(`Serveur Node.js prêt sur le port ${PORT_NODE}`);