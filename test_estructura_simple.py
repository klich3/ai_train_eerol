#!/usr/bin/env python3
"""
🧪 Test específico para crear estructura dental-ai
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_estructura_simple():
    """🧪 Prueba simple de creación de estructura."""
    from Src.workflow_manager import DentalDataWorkflowManager
    
    print("🧪 TEST: CREACIÓN DE ESTRUCTURA DENTAL-AI")
    print("="*50)
    
    # Crear manager
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",
        output_path="Dist/dental_ai"
    )
    
    try:
        print("🏗️ Creando estructura dental-ai...")
        manager.create_dental_ai_structure()
        
        print("✅ Estructura creada exitosamente!")
        
        # Verificar estructura
        base_path = Path("Dist/dental_ai")
        dirs_esperados = ['datasets', 'models', 'training', 'api', 'docs']
        
        print(f"\n🔍 VERIFICANDO ESTRUCTURA:")
        for dir_name in dirs_esperados:
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"✅ {dir_name}/ existe")
            else:
                print(f"❌ {dir_name}/ NO existe o no es directorio")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_estructura_simple()
