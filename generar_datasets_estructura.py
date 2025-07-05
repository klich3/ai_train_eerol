#!/usr/bin/env python3
"""
🏗️ Generador de Datasets en Estructura Correcta
===============================================

Este script convierte tus datasets existentes a la estructura 
que el workflow necesita para generar archivos en la carpeta final.
"""

import sys
import shutil
from pathlib import Path
import json

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def crear_estructura_datasets():
    """Crear estructura de datasets correcta."""
    print("🏗️ GENERADOR DE ESTRUCTURA DE DATASETS")
    print("=" * 50)
    
    base_output = Path("Dist/dental_ai")
    datasets_dir = base_output / "datasets"
    
    # Crear directorios principales
    yolo_dir = datasets_dir / "yolo"
    coco_dir = datasets_dir / "coco"
    unet_dir = datasets_dir / "unet"
    classification_dir = datasets_dir / "classification"
    
    print("📁 Creando estructura de directorios...")
    
    # Crear directorios YOLO
    yolo_subdirs = ["train/images", "train/labels", "val/images", "val/labels", "test/images", "test/labels"]
    for subdir in yolo_subdirs:
        (yolo_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Crear directorios COCO
    coco_subdirs = ["train", "val", "test", "annotations"]
    for subdir in coco_subdirs:
        (coco_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Crear directorios U-Net
    unet_subdirs = ["train/images", "train/masks", "val/images", "val/masks", "test/images", "test/masks"]
    for subdir in unet_subdirs:
        (unet_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Crear directorios de clasificación
    classification_subdirs = ["train", "val", "test"]
    for subdir in classification_subdirs:
        (classification_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios creados")
    
    return yolo_dir, coco_dir, unet_dir, classification_dir

def crear_archivo_yolo_config(yolo_dir):
    """Crear archivo de configuración YOLO."""
    data_yaml = yolo_dir / "data.yaml"
    
    config_content = """# Dental AI Dataset Configuration
# Generado automáticamente

path: .  # Dataset root dir
train: train/images  # Train images (relative to 'path')
val: val/images      # Val images (relative to 'path')
test: test/images    # Test images (optional)

# Classes (categorías dentales detectadas)
names:
  0: caries
  1: tooth
  2: implant
  3: filling
  4: crown
  5: root_canal
  6: bone_loss
  7: impacted
  8: periapical_lesion
  9: maxillary_sinus
  10: mandible
  11: maxilla
"""
    
    with open(data_yaml, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Archivo YOLO config creado: {data_yaml}")

def crear_anotaciones_coco(coco_dir):
    """Crear estructura básica de anotaciones COCO."""
    annotations_dir = coco_dir / "annotations"
    
    # Estructura básica COCO
    coco_structure = {
        "info": {
            "description": "Dental AI Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Smart Dental AI Workflow Manager",
            "date_created": "2025-07-05"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "caries", "supercategory": "dental"},
            {"id": 2, "name": "tooth", "supercategory": "dental"},
            {"id": 3, "name": "implant", "supercategory": "dental"},
            {"id": 4, "name": "filling", "supercategory": "dental"},
            {"id": 5, "name": "crown", "supercategory": "dental"},
            {"id": 6, "name": "root_canal", "supercategory": "dental"},
            {"id": 7, "name": "bone_loss", "supercategory": "dental"},
            {"id": 8, "name": "impacted", "supercategory": "dental"},
            {"id": 9, "name": "periapical_lesion", "supercategory": "dental"},
            {"id": 10, "name": "maxillary_sinus", "supercategory": "dental"},
            {"id": 11, "name": "mandible", "supercategory": "dental"},
            {"id": 12, "name": "maxilla", "supercategory": "dental"}
        ]
    }
    
    # Crear archivos para train, val, test
    for split in ["train", "val", "test"]:
        annotation_file = annotations_dir / f"instances_{split}.json"
        with open(annotation_file, 'w') as f:
            json.dump(coco_structure, f, indent=2)
    
    print(f"📝 Anotaciones COCO creadas en: {annotations_dir}")

def crear_categorias_clasificacion(classification_dir):
    """Crear categorías para clasificación."""
    categories = [
        "caries", "healthy_tooth", "implant", "filling", 
        "crown", "root_canal", "bone_loss", "impacted",
        "periapical_lesion", "orthodontic", "prosthetic", "other"
    ]
    
    for split in ["train", "val", "test"]:
        split_dir = classification_dir / split
        for category in categories:
            category_dir = split_dir / category
            category_dir.mkdir(exist_ok=True)
    
    print(f"📁 Categorías de clasificación creadas en: {classification_dir}")

def copiar_datasets_existentes():
    """Copiar algunos datasets existentes a la nueva estructura."""
    print("\n📦 COPIANDO DATASETS EXISTENTES...")
    
    source_base = Path("_dataSets")
    target_base = Path("Dist/dental_ai/datasets")
    
    copied_count = 0
    
    # Copiar algunos datasets YOLO
    yolo_source = source_base / "_YOLO"
    yolo_target = target_base / "yolo"
    
    if yolo_source.exists():
        # Copiar el primer dataset YOLO disponible
        yolo_datasets = [d for d in yolo_source.iterdir() if d.is_dir()]
        if yolo_datasets:
            source_dataset = yolo_datasets[0]
            print(f"   📁 Copiando dataset YOLO: {source_dataset.name}")
            
            # Copiar imágenes y etiquetas si existen
            for img_file in source_dataset.rglob("*.jpg"):
                if img_file.parent.name in ["images", "train", "val", "test"] or "image" in img_file.parent.name.lower():
                    target_img = yolo_target / "train/images" / img_file.name
                    try:
                        shutil.copy2(img_file, target_img)
                        copied_count += 1
                        if copied_count >= 5:  # Limitar para demo
                            break
                    except Exception as e:
                        continue
    
    print(f"   ✅ {copied_count} archivos copiados como ejemplo")

def crear_archivos_ejemplo():
    """Crear algunos archivos de ejemplo."""
    print("\n📝 CREANDO ARCHIVOS DE EJEMPLO...")
    
    base_dir = Path("Dist/dental_ai/datasets")
    
    # Crear archivo README
    readme_content = """# 📊 Datasets Dental AI

## 📁 Estructura

### 🎯 YOLO (Detección)
- `yolo/train/` - Imágenes y etiquetas de entrenamiento
- `yolo/val/` - Imágenes y etiquetas de validación
- `yolo/data.yaml` - Configuración del dataset

### 🎭 COCO (Segmentación)
- `coco/train/` - Imágenes de entrenamiento
- `coco/annotations/` - Anotaciones en formato COCO
- `coco/val/` - Imágenes de validación

### 🧩 U-Net (Segmentación Médica)
- `unet/train/images/` - Imágenes de entrenamiento
- `unet/train/masks/` - Máscaras de entrenamiento
- `unet/val/` - Datos de validación

### 📂 Clasificación
- `classification/train/[categoria]/` - Imágenes por categoría
- `classification/val/[categoria]/` - Validación por categoría

## 🚀 Uso

Los scripts de entrenamiento en `../scripts/` están configurados 
para usar esta estructura automáticamente.

---
*Generado por Smart Dental AI Workflow Manager v3.0*
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 README creado: {readme_path}")

def main():
    """Función principal."""
    print("🎯 ¿Qué quieres hacer?")
    print("1. 🏗️ Crear estructura completa de datasets")
    print("2. 📦 Solo crear directorios")
    print("3. 📝 Solo crear archivos de configuración")
    print("4. 🔄 Todo lo anterior + copiar ejemplos")
    print("0. ❌ Salir")
    
    choice = input("\n🎯 Selecciona una opción: ").strip()
    
    if choice == "0":
        print("👋 ¡Hasta luego!")
        return
    
    if choice in ["1", "2", "4"]:
        yolo_dir, coco_dir, unet_dir, classification_dir = crear_estructura_datasets()
    
    if choice in ["1", "3", "4"]:
        if 'yolo_dir' in locals():
            crear_archivo_yolo_config(yolo_dir)
            crear_anotaciones_coco(coco_dir)
            crear_categorias_clasificacion(classification_dir)
            crear_archivos_ejemplo()
    
    if choice == "4":
        copiar_datasets_existentes()
    
    print(f"\n✅ COMPLETADO")
    print(f"📁 Estructura creada en: Dist/dental_ai/datasets/")
    print(f"\n🎯 PRÓXIMOS PASOS:")
    print(f"1. Copia tus imágenes a las carpetas correspondientes")
    print(f"2. Ejecuta: python smart_dental_workflow.py")
    print(f"3. Usa la opción 7 para generar scripts")
    print(f"4. ¡Entrena tus modelos!")

if __name__ == "__main__":
    main()
