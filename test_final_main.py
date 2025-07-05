#!/usr/bin/env python3
"""
🧪 Test final para main.py opción 8 y 9
"""

import sys
from pathlib import Path

# Agregar ruta de módulos  
sys.path.append(str(Path(__file__).parent / "Src"))

def test_main_opciones_8_y_9():
    """🧪 Test final de las opciones 8 y 9 de main.py"""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        
        print("🧪 TEST FINAL: MAIN.PY OPCIONES 8 Y 9")
        print("="*45)
        
        # Crear manager
        manager = DentalDataWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        print("✅ DentalDataWorkflowManager inicializado")
        
        # Opción 8: Crear estructura
        print(f"\n🏗️ OPCIÓN 8: Crear estructura dental-ai...")
        manager.create_dental_ai_structure()
        print("✅ OPCIÓN 8 completada exitosamente")
        
        # Opción 9: Generar scripts
        print(f"\n📝 OPCIÓN 9: Generar scripts de entrenamiento...")
        manager.create_training_scripts()
        print("✅ OPCIÓN 9 completada exitosamente")
        
        # Verificación final
        print(f"\n🔍 VERIFICACIÓN FINAL:")
        
        base_path = Path("Dist/dental_ai")
        
        # Verificar estructura principal
        dirs_principales = ['datasets', 'models', 'training', 'api', 'docs']
        estructura_ok = True
        for dir_name in dirs_principales:
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"✅ {dir_name}/")
            else:
                print(f"❌ {dir_name}/ falta")
                estructura_ok = False
        
        # Verificar scripts
        training_scripts = list((base_path / "training").glob("*.py")) + list((base_path / "training").glob("*.sh"))
        if training_scripts:
            print(f"✅ Scripts de entrenamiento: {len(training_scripts)} archivos")
            for script in training_scripts:
                print(f"   📝 {script.name}")
        else:
            print(f"❌ No se encontraron scripts de entrenamiento")
            estructura_ok = False
        
        return estructura_ok
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_opciones_8_y_9()
    if success:
        print("\n🎉 ¡TODAS LAS CORRECCIONES EXITOSAS!")
        print("✅ Las opciones 8 y 9 de main.py funcionan correctamente")
        print("✅ Ya puedes usar main.py sin problemas")
    else:
        print("\n💥 Aún hay algunos problemas por resolver")
