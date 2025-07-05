#!/usr/bin/env python3
"""
🧪 Test Script Generation
Prueba la generación de scripts de entrenamiento
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from Src.script_templates import ScriptTemplateGenerator

def test_script_generation():
    """Prueba la generación de scripts."""
    print("🧪 Probando generación de scripts...")
    
    output_path = Path("Dist/dental_ai")
    generator = ScriptTemplateGenerator(output_path)
    
    try:
        print("📝 Generando script YOLO...")
        generator.create_yolo_training_script("dental_dataset", "detection")
        print("✅ Script YOLO generado exitosamente")
        
        print("📝 Generando script de segmentación...")
        generator.create_segmentation_training_script("dental_dataset", "segmentation")
        print("✅ Script de segmentación generado exitosamente")
        
        print("📝 Generando script de clasificación...")
        generator.create_classification_training_script("dental_dataset", "classification")
        print("✅ Script de clasificación generado exitosamente")
        
        print("📝 Generando template de API...")
        generator.create_api_template()
        print("✅ Template de API generado exitosamente")
        
        print("\n🎉 Todos los scripts se generaron correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_script_generation()
