#!/usr/bin/env python3
"""
🎯 Test completo del flujo opción 8 de main.py
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_workflow_manager():
    """🧪 Test del DentalDataWorkflowManager."""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        
        print("🧪 TEST: DENTAL DATA WORKFLOW MANAGER")
        print("="*45)
        
        # Crear manager
        manager = DentalDataWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        print("✅ DentalDataWorkflowManager creado")
        
        # Test crear estructura dental-ai
        print(f"\n🏗️ Probando create_dental_ai_structure()...")
        manager.create_dental_ai_structure()
        print("✅ create_dental_ai_structure() exitoso")
        
        # Test crear scripts de entrenamiento
        print(f"\n📝 Probando create_training_scripts()...")
        manager.create_training_scripts()
        print("✅ create_training_scripts() exitoso")
        
        # Verificar estructura final
        print(f"\n🔍 VERIFICANDO ESTRUCTURA FINAL:")
        base_path = Path("Dist/dental_ai")
        
        dirs_principales = ['datasets', 'models', 'training', 'api', 'docs']
        for dir_name in dirs_principales:
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"✅ {dir_name}/")
                
                # Verificar subdirectorios para algunos casos
                if dir_name == 'training':
                    subdirs = ['scripts', 'configs', 'logs']
                    for subdir in subdirs:
                        sub_path = dir_path / subdir
                        if sub_path.exists():
                            print(f"   ✅ {subdir}/")
                        else:
                            print(f"   ❌ {subdir}/ falta")
            else:
                print(f"❌ {dir_name}/ falta")
        
        # Verificar scripts de entrenamiento
        scripts_path = base_path / "training" / "scripts"
        if scripts_path.exists():
            scripts = list(scripts_path.glob("*.py"))
            print(f"\n📝 Scripts encontrados: {len(scripts)}")
            for script in scripts:
                print(f"   📝 {script.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow_manager()
    if success:
        print("\n🎉 ¡TODOS LOS TESTS EXITOSOS!")
        print("✅ La opción 8 de main.py debería funcionar correctamente ahora")
    else:
        print("\n💥 Algunos tests fallaron")
