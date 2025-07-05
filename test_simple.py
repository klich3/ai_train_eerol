#!/usr/bin/env python3
"""
Test simple para el smart workflow.
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_simple():
    """Test muy simple."""
    print("🔧 INICIANDO TEST SIMPLE...")
    
    try:
        print("1. Importando módulo...")
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        print("✅ Módulo importado")
        
        print("2. Inicializando manager...")
        manager = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/test_simple"
        )
        print("✅ Manager inicializado")
        
        print("3. Verificando base path...")
        print(f"   Base path: {manager.base_path}")
        print(f"   Existe: {manager.base_path.exists()}")
        
        if manager.base_path.exists():
            subdirs = [d for d in manager.base_path.iterdir() if d.is_dir()]
            print(f"   Subdirectorios: {len(subdirs)}")
            for subdir in subdirs[:3]:
                print(f"     - {subdir.name}")
        
        print("4. Probando analyzer directamente...")
        analysis = manager.analyzer.scan_datasets()
        print(f"✅ Análisis básico completado")
        print(f"   Total datasets: {analysis.get('total_datasets', 0)}")
        
        print("\n🎉 Test simple completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()
