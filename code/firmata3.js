import { WebSocketServer } from 'ws';
import axios from 'axios';
import pkg from 'johnny-five';
const { Board, Sensor, Servo, Pin } = pkg; // Note: On importe Pin au lieu de Stepper

// --- CONFIGURATION ---
const PORT_NODE = 8085;
const URL_PYTHON = 'http://127.0.0.1:5000/predict';

// PINS MOTEURS (ATTENTION : Pins 8, 9, 10, 11 sur Arduino Mega)
const IN1 = 5;
const IN2 = 4;
const IN3 = 3;
const IN4 = 2;
const SERVO_PIN = 6;

const DISTANCE_COLONNE = 1111; 
const COIN_INIT_POS = 70;      
const COIN_RELEASE_POS = 0;    

// RÉGLAGES CAPTEURS
const CALIBRATION_TIME = 2000;
const SENSOR_FREQ = 20;
const DETECTION_THRESHOLD = 350;

// VARIABLES GLOBALES
const ROWS = 6;
const COLS = 7;
const HUMAN = 1;
const AI = -1;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let isAiThinking = false;
let isCalibrating = true;
let isRobotMoving = false; // Sécurité moteur

let sensorsBaselines = [0, 0, 0, 0, 0, 0, 0];
let calibrationSamples = [[], [], [], [], [], [], []];
let triggers = [false, false, false, false, false, false, false];

// --- WEBSOCKET ---
const wss = new WebSocketServer({ port: PORT_NODE });
function broadcast(data) {
    wss.clients.forEach(c => { if(c.readyState===1) c.send(JSON.stringify(data)); });
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
    console.log("🔄 Reset.");
}

// --- ARDUINO ---
const arduino = new Board({ repl: false });

arduino.on("ready", () => {
    console.log("✅ Arduino Connecté (Mode StandardFirmata) !");
    
    // 1. SERVO
    const servo = new Servo(SERVO_PIN);
    servo.to(COIN_INIT_POS);
    console.log("⚙️ Servo prêt.");

    // 2. MOTEUR STEPPER "MANUEL"
    // On initialise les 4 pins en mode SORTIE
    const motorPins = [
        new Pin(IN1), new Pin(IN2), new Pin(IN3), new Pin(IN4)
    ];

    // Séquence d'activation pour 28BYJ-48 (Half-step)
    const stepSequence = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ];

    // Fonction pour faire tourner le moteur manuellement
    // steps: nombre de pas (+ pour avancer, - pour reculer)
    // delay: vitesse (plus petit = plus vite, min 2ms pour JS)
    const moveStepperManual = (steps, delayTime = 3) => {
        return new Promise((resolve) => {
            let count = 0;
            let stepIndex = 0;
            const direction = steps > 0 ? 1 : -1;
            const total = Math.abs(steps);

            const interval = setInterval(() => {
                // Appliquer la séquence aux pins
                const seq = stepSequence[stepIndex];
                for(let i=0; i<4; i++) {
                    seq[i] === 1 ? motorPins[i].high() : motorPins[i].low();
                }

                // Avancer dans la séquence
                stepIndex += direction;
                if (stepIndex >= 8) stepIndex = 0;
                if (stepIndex < 0) stepIndex = 7;

                count++;
                if (count >= total) {
                    clearInterval(interval);
                    // Éteindre les bobines pour pas chauffer
                    motorPins.forEach(p => p.low());
                    resolve();
                }
            }, delayTime);
        });
    };

    const robotPlay = async (targetCol) => {
        if(isRobotMoving) return;
        isRobotMoving = true;

        const stepsToGo = (targetCol + 1) * DISTANCE_COLONNE;
        console.log(`🤖 Moteur (Manuel) : Avance de ${stepsToGo} pas...`);

        // 1. Avancer
        await moveStepperManual(stepsToGo);
        console.log("   📍 Arrivé. Largage !");
        
        // 2. Lacher
        servo.to(COIN_RELEASE_POS);
        await new Promise(r => setTimeout(r, 1000));
        servo.to(COIN_INIT_POS);
        await new Promise(r => setTimeout(r, 500));

        console.log("   🏠 Retour base...");
        
        // 3. Reculer (Pas négatifs)
        await moveStepperManual(-stepsToGo);
        
        console.log("✅ Robot rentré.");
        isRobotMoving = false;
        isAiThinking = false;
    };


    // 3. CAPTEURS
    console.log("⏳ DÉBUT CALIBRATION (2s)...");
    const sensors = [];
    for (let i = 0; i < 7; i++) {
        sensors[i] = new Sensor({ pin: `A${i}`, freq: 20 });
        sensors[i].on("data", async function() {
            const val = this.value;
            if (isCalibrating) {
                calibrationSamples[i].push(val);
                return;
            }
            const diff = Math.abs(val - sensorsBaselines[i]);
            if (diff > DETECTION_THRESHOLD) {
                if (!triggers[i]) {
                    triggers[i] = true; 
                    console.log(`🔻 PASSAGE COL ${i}`);
                    
                    if (isAiThinking || isRobotMoving) return;

                    if (playMove(i, HUMAN)) {
                        broadcast({ couleur: "Jaune", colonne: i.toString() });
                        isAiThinking = true;

                        try {
                            setTimeout(async () => {
                                const response = await axios.post(URL_PYTHON, { board: board });
                                const aiCol = response.data.column;
                                console.log(`🧠 IA choisit Col ${aiCol}`);

                                if (playMove(aiCol, AI)) {
                                    broadcast({ couleur: "Rouge", colonne: aiCol.toString() });
                                    robotPlay(aiCol);
                                }
                            }, 500);
                        } catch (e) { console.error(e); isAiThinking = false; }
                    }
                }
            } else {
                if (diff < (DETECTION_THRESHOLD / 2)) triggers[i] = false;
            }
        });
    }

    setTimeout(() => {
        console.log("📊 Fin Calibration.");
        for (let i = 0; i < 7; i++) {
            const s = calibrationSamples[i];
            if (s.length > 0) sensorsBaselines[i] = Math.floor(s.reduce((a, b) => a + b, 0) / s.length);
        }
        isCalibrating = false;
        console.log("🟢 JEU PRÊT !");
    }, CALIBRATION_TIME);
});

// WEB
wss.on('connection', ws => {
    ws.on('message', msg => { if (JSON.parse(msg).action === "RESET") resetGame(); });
});
console.log(`Serveur prêt port ${PORT_NODE}`);
