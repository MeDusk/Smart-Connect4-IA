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
  const [lastMove, setLastMove] = useState(null); // Pour afficher quel coup vient d'être joué

  const pieceIdCounter = useRef(0);

  useEffect(() => {
    const connection = new WebSocket(url);
    wsRef.current = connection;

    connection.onopen = () => console.log("Connecté au Robot");
    
    connection.onmessage = e => {
      const packet = JSON.parse(e.data);
      
      if (packet.action === "RESET_DONE") {
          setBoard(Array.from({ length: ROWS }, () => Array(COLS).fill(null)));
          setLastMove("Partie réinitialisée");
          return;
      }

      if (packet.couleur && packet.colonne) {
          handlePacket(packet);
      }
    };

    return () => connection.close();
  }, [url]);

  const handlePacket = (packet) => {
    const { couleur, colonne } = packet;
    const colIndex = parseInt(colonne, 10);
    
    setLastMove(`${couleur} a joué en colonne ${colIndex}`);

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
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '20px' }}>
      <h1>Moniteur Robot Puissance 4</h1>
      
      <div style={{ marginBottom: '20px', fontSize: '1.2rem', fontWeight: 'bold', color: '#333' }}>
        Status : {lastMove || "En attente de détection..."}
      </div>

      <div style={{ 
        backgroundColor: '#1e3a8a', 
        padding: '10px', 
        borderRadius: '10px',
        display: 'inline-block',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)'
      }}>
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
      <button onClick={resetGame} style={{ marginTop: '30px', padding: '10px 20px', cursor: 'pointer' }}>
        RESET
      </button>
      
      <div style={{marginTop: '20px', color: '#666', fontSize: '0.8rem'}}>
          Capteurs actifs : A0, A1, A2, A3, A4, A5, A6
      </div>
    </div>
  );
}