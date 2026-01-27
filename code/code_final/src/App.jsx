"use client"

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ROWS = 6;
const COLS = 7;

export default function Puissance4() {
  const [url] = useState("ws://localhost:8085");
  const wsRef = useRef(null);

  const [board, setBoard] = useState(
    Array.from({ length: ROWS }, () => Array(COLS).fill(null))
  );
  const [statusMessage, setStatusMessage] = useState("En attente du robot...");
  const [winner, setWinner] = useState(null); // 'Jaune', 'Rouge' ou 'Nul'

  const pieceIdCounter = useRef(0);

  useEffect(() => {
    const connection = new WebSocket(url);
    wsRef.current = connection;

    connection.onopen = () => {
        console.log("Connecté au Robot");
        setStatusMessage("Prêt ! Jouez un jeton.");
    };

    connection.onmessage = e => {
      const packet = JSON.parse(e.data);

      // RESET
      if (packet.action === "RESET_DONE") {
          setBoard(Array.from({ length: ROWS }, () => Array(COLS).fill(null)));
          setWinner(null);
          setStatusMessage("Nouvelle partie ! À toi de jouer.");
          return;
      }

      // FIN DE PARTIE
      if (packet.action === "GAME_OVER") {
          setWinner(packet.winner);
          if (packet.winner === "Jaune") setStatusMessage("🎉 VICTOIRE HUMAINE !");
          else if (packet.winner === "Rouge") setStatusMessage("🤖 VICTOIRE ROBOT !");
          else setStatusMessage("😐 MATCH NUL.");
          return;
      }

      // COUP JOUÉ
      if (packet.couleur && packet.colonne) {
          handlePacket(packet);
      }
    };

    return () => connection.close();
  }, [url]);

  const handlePacket = (packet) => {
    const { couleur, colonne } = packet;
    const colIndex = parseInt(colonne, 10);

    // Message de statut
    if (couleur === "Jaune") setStatusMessage("L'IA réfléchit...");
    if (couleur === "Rouge") setStatusMessage("À toi de jouer !");

    setBoard((prevBoard) => {
      const newBoard = prevBoard.map(row => [...row]);
      for (let r = ROWS - 1; r >= 0; r--) {
        if (!newBoard[r][colIndex]) {
          newBoard[r][colIndex] = {
            color: couleur,
            id: pieceIdCounter.current++
          };
          break;
        }
      }
      return newBoard;
    });
  };

  const resetGame = () => {
    if (wsRef.current) wsRef.current.send(JSON.stringify({ action: "RESET" }));
  };

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '20px', backgroundColor: '#f0f9ff', minHeight: '100vh' }}>

      <h1 style={{color: '#1e3a8a'}}>Puissance 4 Robotique</h1>

      {/* BANDEAU DE STATUT */}
      <div style={{
          marginBottom: '20px',
          padding: '15px',
          fontSize: '1.5rem',
          fontWeight: 'bold',
          color: winner ? 'white' : '#333',
          backgroundColor: winner === 'Jaune' ? '#eab308' : (winner === 'Rouge' ? '#ef4444' : '#f0f9ff'),
          borderRadius: '8px',
          transition: 'all 0.3s'
      }}>
        {statusMessage}
      </div>

      <div style={{
        backgroundColor: '#1e3a8a',
        padding: '10px',
        borderRadius: '10px',
        display: 'inline-block',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)',
        position: 'relative'
      }}>
        {/* Overlay si fin de partie */}
        {winner && (
            <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: '10px', zIndex: 20,
                display: 'flex', justifyContent: 'center', alignItems: 'center'
            }}>
                <button
                    onClick={resetGame}
                    style={{
                        padding: '15px 30px', fontSize: '1.5rem', fontWeight: 'bold',
                        backgroundColor: '#22c55e', color: 'white', border: 'none', borderRadius: '50px',
                        cursor: 'pointer', boxShadow: '0 5px 15px rgba(0,0,0,0.3)'
                    }}
                >
                    REJOUER ↻
                </button>
            </div>
        )}

        {board.map((row, rowIndex) => (
          <div key={rowIndex} style={{ display: 'flex' }}>
            {row.map((cell, colIndex) => (
              <div
                key={colIndex}
                style={{
                  width: '60px', height: '60px',
                  backgroundColor: '#1e40af',
                  border: '1px solid #172554',
                  position: 'relative', overflow: 'hidden',
                  display: 'flex', justifyContent: 'center', alignItems: 'center'
                }}
              >
                <div style={{ position: 'absolute', width: '50px', height: '50px', borderRadius: '50%', backgroundColor: 'white', zIndex: 0 }} />
                <AnimatePresence>
                  {cell && (
                    <motion.div
                      key={cell.id}
                      initial={{ y: -400, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 200, damping: 15 }}
                      style={{
                        width: '50px', height: '50px', borderRadius: '50%',
                        backgroundColor: cell.color === 'Jaune' ? '#facc15' : '#ef4444',
                        zIndex: 10, boxShadow: 'inset -2px -2px 5px rgba(0,0,0,0.3)'
                      }}
                    />
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
        ))}
      </div>

      <br />
      {!winner && (
        <button onClick={resetGame} style={{ marginTop: '30px', padding: '10px 20px', cursor: 'pointer', border:'1px solid #ccc', borderRadius:'4px' }}>
            Force Reset
        </button>
      )}

      <div style={{marginTop: '20px', color: '#666', fontSize: '0.8rem'}}>
          Capteurs actifs : A0 -> A6
      </div>
    </div>
  );
}
