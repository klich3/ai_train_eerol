#!/usr/bin/env python3
"""
Regenerador de scripts de entrenamiento con rutas corregidas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Src.workflow_manager import DentalDataWorkflowManager

def main():
    print("🔧 Regenerando scripts de entrenamiento...")
    
    # Crear el manager
    manager = DentalDataWorkflowManager()
    
    # Regenerar scripts
    try:
        manager.create_training_scripts()
        print("✅ Scripts de entrenamiento regenerados exitosamente")
    except Exception as e:
        print(f"❌ Error regenerando scripts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
