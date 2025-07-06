#!/usr/bin/env python3
"""
🚀 Quick Model Test
Prueba rápida de modelos YOLO
"""

import os
import sys
from pathlib import Path

# Agregar ruta al proyecto principal
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_model():
    """Buscar modelos entrenados disponibles."""
    base_dir = Path(__file__).parent
    
    # Posibles ubicaciones de modelos
    search_paths = [
        base_dir / "models" / "yolo_detect",
        base_dir / "datasets" / "detection_combined" / "logs",
        base_dir / "training" / "logs",
    ]
    
    models_found = []
    
    for search_path in search_paths:
        if search_path.exists():
            # Buscar archivos .pt
            pt_files = list(search_path.rglob("*.pt"))
            for pt_file in pt_files:
                if "best.pt" in pt_file.name or "last.pt" in pt_file.name:
                    models_found.append(pt_file)
    
    return models_found

def find_test_images():
    """Buscar imágenes de prueba."""
    base_dir = Path(__file__).parent
    
    # Buscar imágenes en el dataset
    test_paths = [
        base_dir / "datasets" / "detection_combined" / "test" / "images",
        base_dir / "datasets" / "detection_combined" / "val" / "images",
        base_dir / "datasets" / "detection_combined" / "train" / "images",
    ]
    
    for test_path in test_paths:
        if test_path.exists():
            images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
            if images:
                return test_path, images[:5]  # Primeras 5 imágenes
    
    return None, []

def quick_test():
    """Ejecutar prueba rápida."""
    print("🔍 Buscando modelos entrenados...")
    
    models = find_model()
    if not models:
        print("❌ No se encontraron modelos entrenados")
        print("💡 Ubicaciones buscadas:")
        print("   - models/yolo_detect/")
        print("   - datasets/detection_combined/logs/")
        print("   - training/logs/")
        print("\n🚀 Para entrenar un modelo, ejecuta:")
        print("   cd datasets/detection_combined")
        print("   ./train_dental_dataset.sh")
        return
    
    print(f"✅ Encontrados {len(models)} modelos:")
    for i, model in enumerate(models):
        print(f"   {i+1}. {model}")
    
    # Usar el primer modelo (generalmente best.pt)
    selected_model = models[0]
    print(f"\n🎯 Usando modelo: {selected_model}")
    
    # Buscar imágenes de prueba
    print("\n🔍 Buscando imágenes de prueba...")
    test_dir, test_images = find_test_images()
    
    if not test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"✅ Encontradas {len(test_images)} imágenes en: {test_dir}")
    
    # Ejecutar prueba
    print(f"\n🧪 Ejecutando prueba con el tester...")
    
    # Importar y usar el tester
    try:
        from test_model import DentalYOLOTester
        
        # Inicializar tester
        dataset_path = Path(__file__).parent / "datasets" / "detection_combined"
        tester = DentalYOLOTester(selected_model, dataset_path)
        
        # Crear directorio de salida
        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        # Probar primera imagen
        print(f"📸 Probando imagen: {test_images[0].name}")
        results = tester.test_single_image(test_images[0], output_dir)
        
        # Probar conjunto pequeño
        print(f"\n📊 Probando conjunto de 3 imágenes...")
        split = "val" if "val" in str(test_dir) else "test" if "test" in str(test_dir) else "train"
        summary = tester.test_dataset(split=split, max_images=3, output_dir=output_dir)
        
        print(f"\n🎉 Prueba completada!")
        print(f"📂 Resultados guardados en: {output_dir}")
        print(f"💡 Para más opciones, usa: python test_model.py --help")
        
    except ImportError as e:
        print(f"❌ Error al importar tester: {e}")
        print("💡 Asegúrate de tener ultralytics instalado:")
        print("   pip install ultralytics")
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")

if __name__ == "__main__":
    print("🚀 QUICK MODEL TEST")
    print("==================")
    quick_test()
