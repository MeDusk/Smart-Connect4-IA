#!/usr/bin/env python3
"""
Script de test pour valider inference.py
"""

import subprocess
import json
import time
import sys

def test_inference_complete():
    """Test complet du script d'inférence"""
    print("="*70)
    print("TEST COMPLET DU SCRIPT D'INFÉRENCE")
    print("="*70)
    
    # Plateau de test
    test_board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0]
    ]
    
    # Lancer le processus
    print("\n⏳ Démarrage du serveur d'inférence...")
    process = subprocess.Popen(
        ['python', 'inference.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # ATTENDRE que le processus soit prêt
    time.sleep(2)  # Laisser le temps au modèle de charger
    
    def read_json_response(timeout=5):
        """Lit une ligne JSON valide depuis stdout avec timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
    
    try:
        # Test 1: Signal ready
        print("\n[1/5] Test du signal ready...")
        ready_msg = read_json_response(timeout=5)
        
        if ready_msg and ready_msg.get('status') == 'ready':
            print(f"   ✓ Status: {ready_msg['status']}")
            print(f"   ✓ Device: {ready_msg['config']['device']}")
            print(f"   ✓ Version: {ready_msg['config']['version']}")
        else:
            print(f"   ✗ Erreur: Signal ready non reçu")
            print(f"   Debug: Réponse = {ready_msg}")
            return False
        
        # Test 2: Prédiction
        print("\n[2/5] Test de prédiction...")
        start = time.time()
        request = {"command": "predict", "board": test_board}
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        
        response = read_json_response(timeout=10)
        elapsed = (time.time() - start) * 1000
        
        if response and response.get('status') == 'success':
            print(f"   ✓ Colonne choisie: {response['column']}")
            print(f"   ✓ Confiance: {response['metadata']['confidence']:.1%}")
            print(f"   ✓ Temps total: {elapsed:.2f}ms")
            print(f"   ✓ Temps inférence: {response['metadata']['inference_time_ms']:.2f}ms")
        else:
            print(f"   ✗ Erreur: {response}")
            return False
        
        # Test 3: Ping
        print("\n[3/5] Test de ping...")
        process.stdin.write(json.dumps({"command": "ping"}) + '\n')
        process.stdin.flush()
        
        ping_response = read_json_response()
        if ping_response and ping_response.get('status') == 'pong':
            print(f"   ✓ Status: {ping_response['status']}")
            print(f"   ✓ Uptime: {ping_response['uptime_seconds']:.2f}s")
        else:
            print(f"   ✗ Erreur ping")
            return False
        
        # Test 4: Stats
        print("\n[4/5] Test de statistiques...")
        process.stdin.write(json.dumps({"command": "stats"}) + '\n')
        process.stdin.flush()
        
        stats_response = read_json_response()
        if stats_response and stats_response.get('status') == 'success':
            print(f"   ✓ Prédictions: {stats_response['stats']['predictions_count']}")
            print(f"   ✓ Temps moyen: {stats_response['stats']['average_inference_time_ms']:.2f}ms")
            print(f"   ✓ Device: {stats_response['stats']['device']}")
        else:
            print(f"   ✗ Erreur stats")
            return False
        
        # Test 5: Shutdown
        print("\n[5/5] Test d'arrêt propre...")
        process.stdin.write(json.dumps({"command": "shutdown"}) + '\n')
        process.stdin.flush()
        
        shutdown_response = read_json_response()
        if shutdown_response and shutdown_response.get('status') == 'shutdown':
            print(f"   ✓ {shutdown_response['message']}")
        else:
            print(f"   ✗ Erreur shutdown (non-bloquant)")
        
        # Attendre la fin
        process.wait(timeout=3)
        
        print("\n" + "="*70)
        print("✓✓✓ TOUS LES TESTS RÉUSSIS ✓✓✓")
        print("="*70)
        print("\n🎉 Module d'inférence validé et prêt pour production !")
        return True
        
    except Exception as e:
        print(f"\n✗ Erreur pendant les tests: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except:
                process.kill()

if __name__ == "__main__":
    success = test_inference_complete()
    sys.exit(0 if success else 1)
