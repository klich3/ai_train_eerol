#!/usr/bin/env python3
"""
🧪 TEST DE RUTAS - Scripts de Entrenamiento
==========================================

Prueba que las rutas de los scripts estén correctas sin ejecutar el entrenamiento.
"""

import os
import sys
from pathlib import Path

def test_classification_paths():
    """Prueba las rutas del script de clasificación."""
    print("\n🧪 TEST: Rutas de Clasificación")
    print("-" * 40)
    
    data_dir = "../datasets/classification"
    
    if os.path.exists(data_dir):
        print(f"✅ Dataset principal encontrado: {data_dir}")
        
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        if os.path.exists(train_dir):
            print(f"✅ Directorio train: {train_dir}")
            # Contar subdirectorios (clases)
            classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"📊 Clases encontradas: {len(classes)} - {classes[:3]}...")
        else:
            print(f"❌ Directorio train no encontrado: {train_dir}")
            
        if os.path.exists(val_dir):
            print(f"✅ Directorio val: {val_dir}")
        else:
            print(f"❌ Directorio val no encontrado: {val_dir}")
            
    else:
        print(f"❌ Dataset no encontrado: {data_dir}")

def test_segmentation_paths():
    """Prueba las rutas del script de segmentación."""
    print("\n🧪 TEST: Rutas de Segmentación")
    print("-" * 40)
    
    dataset_path = "../datasets/segmentation_coco"
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset principal encontrado: {dataset_path}")
        
        # Verificar estructura
        annotations_dir = os.path.join(dataset_path, "annotations")
        images_dir = os.path.join(dataset_path, "images")
        
        if os.path.exists(annotations_dir):
            print(f"✅ Directorio annotations: {annotations_dir}")
            # Buscar archivos de anotaciones
            json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            print(f"📊 Archivos de anotaciones: {len(json_files)} - {json_files[:2]}...")
        else:
            print(f"❌ Directorio annotations no encontrado: {annotations_dir}")
            
        if os.path.exists(images_dir):
            print(f"✅ Directorio images: {images_dir}")
            # Contar imágenes
            if os.path.isdir(images_dir):
                images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📊 Imágenes encontradas: {len(images)}")
        else:
            print(f"❌ Directorio images no encontrado: {images_dir}")
            
    else:
        print(f"❌ Dataset no encontrado: {dataset_path}")

def test_yolo_paths():
    """Prueba las rutas del script YOLO."""
    print("\n🧪 TEST: Rutas de YOLO")
    print("-" * 40)
    
    data_yaml = "../datasets/detection_combined/data.yaml"
    
    if os.path.exists(data_yaml):
        print(f"✅ Archivo data.yaml encontrado: {data_yaml}")
        
        # Leer contenido del archivo YAML
        try:
            with open(data_yaml, 'r') as f:
                content = f.read()
                print("📝 Contenido del data.yaml:")
                print(content[:200] + "..." if len(content) > 200 else content)
        except Exception as e:
            print(f"⚠️ Error leyendo data.yaml: {e}")
            
    else:
        print(f"❌ Archivo data.yaml no encontrado: {data_yaml}")
        
        # Verificar directorio base
        dataset_dir = "../datasets/detection_combined"
        if os.path.exists(dataset_dir):
            print(f"✅ Directorio del dataset: {dataset_dir}")
            files = os.listdir(dataset_dir)
            print(f"📁 Archivos disponibles: {files}")
        else:
            print(f"❌ Directorio del dataset no encontrado: {dataset_dir}")

def main():
    print("🧪 TEST DE RUTAS - Scripts de Entrenamiento")
    print("=" * 50)
    
    # Cambiar al directorio correcto
    script_dir = Path(__file__).parent / "Dist" / "dental_ai" / "training"
    os.chdir(script_dir)
    print(f"📁 Directorio de trabajo: {os.getcwd()}")
    
    # Ejecutar tests
    test_classification_paths()
    test_segmentation_paths()
    test_yolo_paths()
    
    print("\n" + "=" * 50)
    print("✅ Test de rutas completado")

if __name__ == "__main__":
    main()
