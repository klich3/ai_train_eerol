#!/usr/bin/env python3
"""
🧪 Test específico para generar scripts de entrenamiento
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_script_generation():
    """🧪 Test solo para generación de scripts."""
    try:
        from Src.script_templates import ScriptTemplateGenerator
        
        print("🧪 TEST: GENERACIÓN DE SCRIPTS")
        print("="*35)
        
        output_path = Path("Dist/dental_ai")
        generator = ScriptTemplateGenerator(output_path)
        print("✅ ScriptTemplateGenerator creado")
        
        # Crear directorio de entrenamiento si no existe
        training_dir = output_path / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio training verificado: {training_dir}")
        
        # Test 1: YOLO
        print(f"\n📝 Generando script YOLO...")
        generator.create_yolo_training_script("dental_dataset", "detection")
        print("✅ Script YOLO generado")
        
        # Test 2: Segmentación
        print(f"\n📝 Generando script segmentación...")
        generator.create_segmentation_training_script("dental_dataset", "segmentation")
        print("✅ Script segmentación generado")
        
        # Test 3: Clasificación
        print(f"\n📝 Generando script clasificación...")
        generator.create_classification_training_script("dental_dataset", "classification")
        print("✅ Script clasificación generado")
        
        # Verificar archivos creados
        print(f"\n🔍 VERIFICANDO ARCHIVOS CREADOS:")
        scripts_created = list(training_dir.glob("*.py")) + list(training_dir.glob("*.sh"))
        for script in scripts_created:
            print(f"   📝 {script.name}")
        
        print(f"\n📊 Total scripts creados: {len(scripts_created)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_script_generation()
    if success:
        print("\n🎉 ¡GENERACIÓN DE SCRIPTS EXITOSA!")
    else:
        print("\n💥 Error en generación de scripts")
