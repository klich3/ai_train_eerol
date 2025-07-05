#!/usr/bin/env python3
"""
🧪 Test directo de main.py opción 8
"""

import subprocess
import sys

def test_main_opcion_8():
    """🧪 Prueba la opción 8 de main.py"""
    print("🧪 PROBANDO MAIN.PY OPCIÓN 8...")
    print("="*40)
    
    try:
        # Ejecutar main.py con entrada automática
        cmd = ['python', 'main.py']
        
        # Simular entrada del usuario: opción 8, confirmar con 's', luego salir con '0'
        entrada = "8\ns\n0\n"
        
        process = subprocess.run(
            cmd, 
            input=entrada, 
            text=True, 
            capture_output=True, 
            timeout=30
        )
        
        print("📤 SALIDA STDOUT:")
        print(process.stdout)
        
        if process.stderr:
            print("📤 SALIDA STDERR:")
            print(process.stderr)
        
        print(f"📊 CÓDIGO DE SALIDA: {process.returncode}")
        
        return process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT: El proceso tardó demasiado")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_main_opcion_8()
    if success:
        print("✅ Test completado exitosamente")
    else:
        print("❌ Test falló")
