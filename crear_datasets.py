#!/usr/bin/env python3
"""
📦 Creador de Datasets para Dental AI Workflow
==============================================

Script para crear/organizar datasets en la estructura correcta
para que aparezcan en la carpeta final después del procesamiento.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def crear_estructura_yolo_ejemplo():
    """Crear estructura de ejemplo para YOLO."""
    print("🎯 CREANDO DATASET YOLO DE EJEMPLO...")
    
    # Crear directorio base
    dataset_path = Path("Dist/dental_ai/datasets/yolo")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Crear archivo de configuración data.yaml
    data_yaml_content = """# Dental AI YOLO Dataset Configuration
# Generado automáticamente

path: .  # Dataset root dir
train: train/images  # Train images (relative to 'path')
val: val/images      # Val images (relative to 'path')
test: test/images    # Test images (relative to 'path')

# Classes
names:
  0: caries
  1: tooth
  2: implant
  3: crown
  4: filling
  5: root_canal
  6: bone_loss
  7: impacted
  8: maxillary_sinus
  9: mandible
  10: maxilla
  11: periapical_lesion
"""
    
    (dataset_path / "data.yaml").write_text(data_yaml_content)
    
    # Crear archivo README
    readme_content = f"""# Dataset YOLO - Dental AI

Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Estructura
```
yolo/
├── data.yaml          # Configuración del dataset
├── train/
│   ├── images/        # Imágenes de entrenamiento
│   └── labels/        # Etiquetas YOLO (.txt)
├── val/
│   ├── images/        # Imágenes de validación
│   └── labels/        # Etiquetas de validación
└── test/
    ├── images/        # Imágenes de prueba
    └── labels/        # Etiquetas de prueba
```

## Uso
```bash
python train_yolo.py
```

## Clases
- 0: caries
- 1: tooth
- 2: implant
- 3: crown
- 4: filling
- 5: root_canal
- 6: bone_loss
- 7: impacted
- 8: maxillary_sinus
- 9: mandible
- 10: maxilla
- 11: periapical_lesion
"""
    
    (dataset_path / "README.md").write_text(readme_content)
    
    print(f"✅ Dataset YOLO creado en: {dataset_path}")
    return dataset_path

def crear_estructura_coco_ejemplo():
    """Crear estructura de ejemplo para COCO."""
    print("🎭 CREANDO DATASET COCO DE EJEMPLO...")
    
    # Crear directorio base
    dataset_path = Path("Dist/dental_ai/datasets/coco")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        (dataset_path / split).mkdir(exist_ok=True)
    
    (dataset_path / 'annotations').mkdir(exist_ok=True)
    
    # Crear archivos de anotaciones COCO básicos
    coco_template = {
        "info": {
            "description": "Dental AI Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Smart Dental AI Workflow Manager",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Academic Use Only",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "caries", "supercategory": "dental"},
            {"id": 2, "name": "tooth", "supercategory": "dental"},
            {"id": 3, "name": "implant", "supercategory": "dental"},
            {"id": 4, "name": "crown", "supercategory": "dental"},
            {"id": 5, "name": "filling", "supercategory": "dental"},
            {"id": 6, "name": "root_canal", "supercategory": "dental"},
            {"id": 7, "name": "bone_loss", "supercategory": "dental"},
            {"id": 8, "name": "impacted", "supercategory": "dental"},
            {"id": 9, "name": "maxillary_sinus", "supercategory": "dental"},
            {"id": 10, "name": "mandible", "supercategory": "dental"},
            {"id": 11, "name": "maxilla", "supercategory": "dental"},
            {"id": 12, "name": "periapical_lesion", "supercategory": "dental"}
        ]
    }
    
    import json
    
    # Crear archivos de anotaciones para cada split
    for split in ['train', 'val', 'test']:
        annotation_file = dataset_path / 'annotations' / f'instances_{split}.json'
        with open(annotation_file, 'w') as f:
            json.dump(coco_template, f, indent=2)
    
    # Crear README
    readme_content = f"""# Dataset COCO - Dental AI

Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Estructura
```
coco/
├── annotations/
│   ├── instances_train.json    # Anotaciones de entrenamiento
│   ├── instances_val.json      # Anotaciones de validación
│   └── instances_test.json     # Anotaciones de prueba
├── train/                      # Imágenes de entrenamiento
├── val/                        # Imágenes de validación
└── test/                       # Imágenes de prueba
```

## Uso
```bash
python train_coco.py
```

## Categorías
{chr(10).join([f"- {cat['id']}: {cat['name']}" for cat in coco_template['categories']])}
"""
    
    (dataset_path / "README.md").write_text(readme_content)
    
    print(f"✅ Dataset COCO creado en: {dataset_path}")
    return dataset_path

def crear_estructura_unet_ejemplo():
    """Crear estructura de ejemplo para U-Net."""
    print("🧩 CREANDO DATASET U-NET DE EJEMPLO...")
    
    # Crear directorio base
    dataset_path = Path("Dist/dental_ai/datasets/unet")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorios
    for split in ['train', 'val', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Crear README
    readme_content = f"""# Dataset U-Net - Dental AI

Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Estructura
```
unet/
├── train/
│   ├── images/        # Imágenes de entrenamiento
│   └── masks/         # Máscaras de segmentación
├── val/
│   ├── images/        # Imágenes de validación
│   └── masks/         # Máscaras de validación
└── test/
    ├── images/        # Imágenes de prueba
    └── masks/         # Máscaras de prueba
```

## Uso
```bash
python train_unet.py
```

## Formato de Máscaras
- Formato: PNG en escala de grises
- Valores: 0 (fondo), 255 (objeto)
- Misma resolución que la imagen original
"""
    
    (dataset_path / "README.md").write_text(readme_content)
    
    print(f"✅ Dataset U-Net creado en: {dataset_path}")
    return dataset_path

def crear_estructura_clasificacion_ejemplo():
    """Crear estructura de ejemplo para clasificación."""
    print("📂 CREANDO DATASET CLASIFICACIÓN DE EJEMPLO...")
    
    # Crear directorio base
    dataset_path = Path("Dist/dental_ai/datasets/classification")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Clases dentales
    clases = ['caries', 'tooth_healthy', 'implant', 'crown', 'filling', 'root_canal']
    
    # Crear subdirectorios para cada split y clase
    for split in ['train', 'val', 'test']:
        for clase in clases:
            (dataset_path / split / clase).mkdir(parents=True, exist_ok=True)
    
    # Crear README
    readme_content = f"""# Dataset Clasificación - Dental AI

Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Estructura
```
classification/
├── train/
│   ├── caries/           # Imágenes con caries
│   ├── tooth_healthy/    # Dientes sanos
│   ├── implant/          # Implantes
│   ├── crown/            # Coronas
│   ├── filling/          # Empastes
│   └── root_canal/       # Tratamientos de conducto
├── val/
│   ├── caries/
│   ├── tooth_healthy/
│   └── ...
└── test/
    ├── caries/
    ├── tooth_healthy/
    └── ...
```

## Uso
```bash
python train_classification.py
```

## Clases
{chr(10).join([f"- {clase}" for clase in clases])}
"""
    
    (dataset_path / "README.md").write_text(readme_content)
    
    print(f"✅ Dataset de clasificación creado en: {dataset_path}")
    return dataset_path

def convertir_datasets_existentes():
    """Convertir datasets existentes a las nuevas estructuras."""
    print("🔄 CONVIRTIENDO DATASETS EXISTENTES...")
    
    datasets_source = Path("Dist/dental_ai/datasets")
    if not datasets_source.exists():
        print("❌ No se encontró el directorio de datasets")
        return
    
    # Mapear datasets existentes a nuevos formatos
    mapeo = {
        'detection_combined': 'yolo',
        'segmentation_coco': 'coco', 
        'segmentation_bitmap': 'unet',
        'classification': 'classification'
    }
    
    datasets_convertidos = []
    
    for old_name, new_name in mapeo.items():
        old_path = datasets_source / old_name
        new_path = datasets_source / new_name
        
        if old_path.exists() and not new_path.exists():
            print(f"   📦 Convirtiendo {old_name} → {new_name}")
            
            # Crear copia con nuevo nombre
            shutil.copytree(old_path, new_path)
            datasets_convertidos.append((old_name, new_name))
            
            # Crear archivo de configuración específico si es YOLO
            if new_name == 'yolo':
                crear_config_yolo(new_path)
    
    if datasets_convertidos:
        print(f"✅ Convertidos {len(datasets_convertidos)} datasets:")
        for old, new in datasets_convertidos:
            print(f"   • {old} → {new}")
    else:
        print("ℹ️ No hay datasets para convertir")

def crear_config_yolo(dataset_path):
    """Crear configuración YOLO para dataset convertido."""
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        config_content = """# Dental AI YOLO Dataset Configuration
path: .
train: train
val: val
test: test

names:
  0: caries
  1: tooth
  2: implant
  3: crown
  4: filling
  5: root_canal
  6: bone_loss
  7: impacted
  8: maxillary_sinus
  9: mandible
  10: maxilla
  11: periapical_lesion
"""
        data_yaml.write_text(config_content)
        print(f"   📝 Configuración YOLO creada: {data_yaml}")

def verificar_estructura_final():
    """Verificar la estructura final de datasets."""
    print("\n📊 VERIFICANDO ESTRUCTURA FINAL...")
    
    datasets_dir = Path("Dist/dental_ai/datasets")
    
    if not datasets_dir.exists():
        print("❌ Directorio datasets no existe")
        return
    
    formatos_esperados = ['yolo', 'coco', 'unet', 'classification']
    datasets_encontrados = []
    
    for formato in formatos_esperados:
        formato_path = datasets_dir / formato
        if formato_path.exists():
            print(f"   ✅ {formato}/")
            datasets_encontrados.append(formato)
            
            # Mostrar contenido
            contenido = [item.name for item in formato_path.iterdir() if item.is_dir()]
            if contenido:
                print(f"      Contiene: {', '.join(contenido)}")
        else:
            print(f"   ❌ {formato}/")
    
    print(f"\n📈 Resumen:")
    print(f"   Formatos disponibles: {len(datasets_encontrados)}/4")
    print(f"   Datasets listos para scripts: {datasets_encontrados}")
    
    return datasets_encontrados

def main():
    """Función principal."""
    print("📦 CREADOR DE DATASETS - DENTAL AI WORKFLOW")
    print("=" * 60)
    print()
    
    print("¿Qué quieres hacer?")
    print("1. 🏗️ Crear estructuras de ejemplo (vacías)")
    print("2. 🔄 Convertir datasets existentes")
    print("3. ✅ Verificar estructura actual")
    print("4. 🚀 Todo (crear + convertir + verificar)")
    print()
    
    choice = input("🎯 Selecciona una opción (1-4): ").strip()
    
    if choice in ['1', '4']:
        print("\n🏗️ CREANDO ESTRUCTURAS DE EJEMPLO...")
        crear_estructura_yolo_ejemplo()
        crear_estructura_coco_ejemplo()
        crear_estructura_unet_ejemplo()
        crear_estructura_clasificacion_ejemplo()
    
    if choice in ['2', '4']:
        print("\n🔄 CONVIRTIENDO DATASETS EXISTENTES...")
        convertir_datasets_existentes()
    
    if choice in ['3', '4']:
        print("\n✅ VERIFICANDO ESTRUCTURA...")
        datasets_disponibles = verificar_estructura_final()
        
        if datasets_disponibles:
            print(f"\n🎉 ¡Perfecto! Ahora puedes usar la opción 7 del workflow")
            print(f"📝 Scripts disponibles para: {', '.join(datasets_disponibles)}")
        else:
            print(f"\n⚠️ No hay datasets en formato correcto")
            print(f"💡 Ejecuta primero las opciones 1 o 2")
    
    print("\n✨ PRÓXIMOS PASOS:")
    print("1. 📝 Ejecuta: python smart_dental_workflow.py")
    print("2. 🎯 Selecciona opción 7: Generar scripts de entrenamiento")
    print("3. 🚀 Los scripts estarán en: Dist/dental_ai/scripts/")

if __name__ == "__main__":
    main()
