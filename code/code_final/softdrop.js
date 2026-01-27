import { WebSocketServer } from 'ws';
import axios from 'axios';
import pkg from 'johnny-five';
const { Board, Sensor, Servo, Pin } = pkg;

// ==========================================
// CONFIGURATION
// ==========================================
const PORT_NODE = 8085;
const URL_PYTHON = 'http://127.0.0.1:5000/predict';

// PINS
const IN1 = 5;
const IN2 = 3;
const IN3 = 4;
const IN4 = 2;
const SERVO_PIN = 6;

// REGLAGES MECANIQUES
const DISTANCE_COLONNE = 1111;
const COIN_INIT_POS = 70;
const COIN_RELEASE_POS = 0;
const STEPPER_DELAY = 4;
const FACTEUR_DETECTION = 0.85;

// ==========================================
// ETAT
// ==========================================
const ROWS = 6;
const COLS = 7;
const HUMAN = 1;
const AI = -1;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let isAiThinking = false;
let isCalibrating = true;
let isRobotMoving = false;
let isGameOver = false; // <--- NOUVEAU

let calibrations = [0, 0, 0, 0, 0, 0, 0];
let calibrationSamples = [[], [], [], [], [], [], []];
let triggers = [false, false, false, false, false, false, false];

// ==========================================
// WEBSOCKET
// ==========================================
const wss = new WebSocketServer({ port: PORT_NODE });
function broadcast(data) {
    wss.clients.forEach(c => { if(c.readyState===1) c.send(JSON.stringify(data)); });
}

// --- LOGIQUE METIER ---
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

// <--- NOUVELLE FONCTION : DETECTION VICTOIRE --->
function checkWin(player) {
    // Horizontal
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            if (board[r][c] === player && board[r][c+1] === player && board[r][c+2] === player && board[r][c+3] === player) return true;
        }
    }
    // Vertical
    for (let r = 0; r < ROWS - 3; r++) {
        for (let c = 0; c < COLS; c++) {
            if (board[r][c] === player && board[r+1][c] === player && board[r+2][c] === player && board[r+3][c] === player) return true;
        }
    }
    // Diagonales
    for (let r = 0; r < ROWS - 3; r++) {
        for (let c = 0; c < COLS - 3; c++) {
            if (board[r][c] === player && board[r+1][c+1] === player && board[r+2][c+2] === player && board[r+3][c+3] === player) return true;
            if (board[r+3][c] === player && board[r+2][c+1] === player && board[r+1][c+2] === player && board[r][c+3] === player) return true;
        }
    }
    return false;
}

// <--- NOUVELLE FONCTION : DETECTION MATCH NUL --->
function checkDraw() {
    return board[0].every(cell => cell !== 0);
}

function resetGame() {
    board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
    isAiThinking = false;
    isRobotMoving = false;
    isGameOver = false; // On débloque le jeu
    broadcast({ action: "RESET_DONE" });
    console.log("🔄 Reset Partie.");
}

// ==========================================
// ARDUINO
// ==========================================
const arduino = new Board({ repl: false });

arduino.on("ready", () => {
    console.log("✅ Arduino Connecté !");

    // 1. SERVO
    const servo = new Servo({
        pin: SERVO_PIN,
        startAt: 90,
        range: [0, 180]
    });
    console.log("🧹 TEST SERVO...");
    servo.sweep();

    // 2. STABILISATION
    arduino.wait(3000, () => {
        console.log("🛑 Fin test Servo.");
        servo.stop();
        servo.to(COIN_INIT_POS);
        console.log("🚀 Initialisation...");
        initSystem(servo);
    });
});

function initSystem(servo) {

    // MOTEUR
    const motorPins = [new Pin(IN1), new Pin(IN2), new Pin(IN3), new Pin(IN4)];
    motorPins.forEach(p => p.low());

    const stepSequence = [
        [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0],
        [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]
    ];

    const moveStepperManual = (steps) => {
        return new Promise((resolve) => {
            let count = 0;
            let stepIndex = 0;
            const direction = steps > 0 ? 1 : -1;
            const total = Math.abs(steps);
            const interval = setInterval(() => {
                const seq = stepSequence[stepIndex];
                for(let i=0; i<4; i++) seq[i] === 1 ? motorPins[i].high() : motorPins[i].low();
                stepIndex += direction;
                if (stepIndex >= 8) stepIndex = 0; if (stepIndex < 0) stepIndex = 7;
                count++;
                if (count >= total) {
                    clearInterval(interval);
                    motorPins.forEach(p => p.low());
                    resolve();
                }
            }, STEPPER_DELAY);
        });
    };

    const softDrop = async () => {
        console.log("   🔓 Largage progressif...");
        for (let pos = COIN_INIT_POS; pos >= COIN_RELEASE_POS; pos -= 5) {
            servo.to(pos);
            await new Promise(r => setTimeout(r, 50));
        }
        await new Promise(r => setTimeout(r, 800));
        console.log("   🔒 Fermeture");
        servo.to(COIN_INIT_POS);
        await new Promise(r => setTimeout(r, 500));
    };

    const robotPlay = async (targetCol) => {
        if(isRobotMoving) return;
        isRobotMoving = true;

        const stepsToGo = (targetCol + 1) * DISTANCE_COLONNE;

        console.log(`🤖 ROBOT : Aller vers Col ${targetCol}`);
        await moveStepperManual(stepsToGo);

        await softDrop();

        console.log("   🏠 Retour");
        await moveStepperManual(-stepsToGo);

        console.log("✅ Robot rentré.");
        isRobotMoving = false;

        // APRES LE MOUVEMENT DU ROBOT : VERIFICATION VICTOIRE IA
        if (checkWin(AI)) {
            console.log("🏆 ROBOT A GAGNÉ !");
            broadcast({ action: "GAME_OVER", winner: "Rouge" });
            isGameOver = true;
        } else if (checkDraw()) {
            broadcast({ action: "GAME_OVER", winner: "Nul" });
            isGameOver = true;
        }

        isAiThinking = false;
    };

    // CAPTEURS
    console.log("⏳ CALIBRATION (2s)...");
    const sensorMapping = ["A6", "A5", "A4", "A3", "A2", "A1", "A0"];
    const sensors = [];

    for (let i = 0; i < 7; i++) {
        sensors[i] = new Sensor({ pin: sensorMapping[i], freq: 20 });

        sensors[i].on("data", async function() {
            const val = this.value;
            if (isCalibrating) {
                calibrationSamples[i].push(val);
                return;
            }

            const seuil = calibrations[i] * FACTEUR_DETECTION;

            if (val < seuil) {
                if (!triggers[i]) {
                    triggers[i] = true;

                    // SI LE JEU EST FINI, ON IGNORE LES CAPTEURS
                    if (isGameOver) {
                        console.log("🚫 Jeu fini, en attente de Reset.");
                        return;
                    }
                    if (isAiThinking || isRobotMoving) return;

                    console.log(`🔻 JOUEUR JOUE COL ${i}`);

                    if (playMove(i, HUMAN)) {
                        broadcast({ couleur: "Jaune", colonne: i.toString() });
                        isAiThinking = true;

                        // VERIFICATION VICTOIRE HUMAIN AVANT D'APPELER L'IA
                        if (checkWin(HUMAN)) {
                            console.log("🏆 HUMAIN A GAGNÉ !");
                            broadcast({ action: "GAME_OVER", winner: "Jaune" });
                            isGameOver = true;
                            // On n'appelle pas l'IA, c'est fini
                            return;
                        }
                        if (checkDraw()) {
                             broadcast({ action: "GAME_OVER", winner: "Nul" });
                             isGameOver = true;
                             return;
                        }

                        // SI PAS GAGNÉ, ON APPELLE L'IA
                        try {
                            setTimeout(async () => {
                                const response = await axios.post(URL_PYTHON, { board: board });
                                const aiCol = response.data.column;
                                console.log(`🧠 IA choisit Col ${aiCol}`);

                                if (playMove(aiCol, AI)) {
                                    broadcast({ couleur: "Rouge", colonne: aiCol.toString() });
                                    robotPlay(aiCol);
                                    // La vérif victoire IA se fait à la fin de robotPlay
                                }
                            }, 500);
                        } catch (e) { console.error(e); isAiThinking = false; }
                    }
                }
            } else {
                if (val > (seuil + 50)) triggers[i] = false;
            }
        });
    }

    setTimeout(() => {
        console.log("📊 Fin Calibration.");
        for (let i = 0; i < 7; i++) {
            const samples = calibrationSamples[i];
            if (samples.length > 0) {
                const sum = samples.reduce((a, b) => a + b, 0);
                calibrations[i] = Math.floor(sum / samples.length);
                console.log(`   Col ${i}: Base = ${calibrations[i]}`);
            }
        }
        isCalibrating = false;
        console.log("🟢 JEU PRÊT !");
    }, 2000);
}

// WEB
wss.on('connection', ws => {
    ws.on('message', msg => { if (JSON.parse(msg).action === "RESET") resetGame(); });
});
console.log(`Serveur prêt port ${PORT_NODE}`);
