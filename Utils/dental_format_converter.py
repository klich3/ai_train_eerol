#!/usr/bin/env python3
"""
Conversor de anotaciones dentales a formatos YOLO y COCO
Este script convierte archivos CSV con anotaciones a formatos compatibles con YOLO y COCO
"""

import os
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class DentalAnnotationConverter:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.class_mapping = {
            'Implant': 0,
            'Fillings': 1
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        
    def csv_to_yolo(self, output_dir="dental_yolo"):
        """Convierte anotaciones CSV a formato YOLO"""
        output_path = self.base_path / output_dir
        output_path.mkdir(exist_ok=True)
        
        print("🔄 Iniciando conversión a formato YOLO...")
        
        # Crear carpetas de salida
        for split in ['train', 'valid', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Procesar cada split
        for split in ['train', 'valid', 'test']:
            split_path = self.base_path / split
            csv_file = split_path / "_annotations.csv"
            
            if not csv_file.exists():
                print(f"⚠️  No se encontró {csv_file}")
                continue
                
            print(f"📁 Procesando {split}...")
            
            # Agrupar anotaciones por imagen
            annotations_by_image = defaultdict(list)
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['filename']
                    width = int(row['width'])
                    height = int(row['height'])
                    class_name = row['class']
                    xmin = int(row['xmin'])
                    ymin = int(row['ymin'])
                    xmax = int(row['xmax'])
                    ymax = int(row['ymax'])
                    
                    # Convertir a formato YOLO (normalizado)
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height
                    
                    class_id = self.class_mapping.get(class_name, 0)
                    
                    annotations_by_image[filename].append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': box_width,
                        'height': box_height
                    })
            
            # Copiar imágenes y crear archivos de labels
            for filename, annotations in annotations_by_image.items():
                # Copiar imagen
                src_img = split_path / filename
                dst_img = output_path / split / 'images' / filename
                
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                    
                    # Crear archivo de label
                    label_filename = filename.replace('.jpg', '.txt')
                    label_path = output_path / split / 'labels' / label_filename
                    
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                                   f"{ann['width']:.6f} {ann['height']:.6f}\n")
        
        # Crear archivo de configuración YOLO
        self._create_yolo_config(output_path)
        print(f"✅ Conversión YOLO completada en: {output_path}")
        
    def csv_to_coco(self, output_dir="dental_coco"):
        """Convierte anotaciones CSV a formato COCO"""
        output_path = self.base_path / output_dir
        output_path.mkdir(exist_ok=True)
        
        print("🔄 Iniciando conversión a formato COCO...")
        
        # Procesar cada split
        for split in ['train', 'valid', 'test']:
            split_path = self.base_path / split
            csv_file = split_path / "_annotations.csv"
            
            if not csv_file.exists():
                print(f"⚠️  No se encontró {csv_file}")
                continue
                
            print(f"📁 Procesando {split}...")
            
            # Crear carpeta para imágenes
            images_dir = output_path / split
            images_dir.mkdir(exist_ok=True)
            
            # Estructura COCO
            coco_data = {
                "info": {
                    "description": "Dental Radiography Dataset",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().isoformat()
                },
                "categories": [
                    {"id": 0, "name": "Implant", "supercategory": "dental"},
                    {"id": 1, "name": "Fillings", "supercategory": "dental"}
                ],
                "images": [],
                "annotations": []
            }
            
            image_id = 0
            annotation_id = 0
            image_mapping = {}
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['filename']
                    width = int(row['width'])
                    height = int(row['height'])
                    class_name = row['class']
                    xmin = int(row['xmin'])
                    ymin = int(row['ymin'])
                    xmax = int(row['xmax'])
                    ymax = int(row['ymax'])
                    
                    # Agregar imagen si no existe
                    if filename not in image_mapping:
                        # Copiar imagen
                        src_img = split_path / filename
                        dst_img = images_dir / filename
                        
                        if src_img.exists():
                            shutil.copy2(src_img, dst_img)
                        
                        coco_data["images"].append({
                            "id": image_id,
                            "file_name": filename,
                            "width": width,
                            "height": height
                        })
                        
                        image_mapping[filename] = image_id
                        image_id += 1
                    
                    # Agregar anotación
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    area = bbox_width * bbox_height
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_mapping[filename],
                        "category_id": self.class_mapping.get(class_name, 0),
                        "bbox": [xmin, ymin, bbox_width, bbox_height],
                        "area": area,
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
            
            # Guardar archivo COCO JSON
            coco_file = output_path / f"{split}.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
        
        print(f"✅ Conversión COCO completada en: {output_path}")
    
    def _create_yolo_config(self, output_path):
        """Crea archivo de configuración para YOLO"""
        config = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        # Crear data.yaml
        yaml_content = f"""# Dental Radiography Dataset Configuration
path: {output_path.absolute()}
train: train/images
val: valid/images
test: test/images

# Classes
nc: {len(self.class_mapping)}
names: {list(self.class_mapping.keys())}
"""
        
        with open(output_path / "data.yaml", 'w') as f:
            f.write(yaml_content)
    
    def create_summary_report(self):
        """Genera un reporte resumen del dataset"""
        report = {
            'dataset_name': 'Dental Radiography',
            'splits': {},
            'classes': self.class_mapping,
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int)
        }
        
        for split in ['train', 'valid', 'test']:
            split_path = self.base_path / split
            csv_file = split_path / "_annotations.csv"
            
            if not csv_file.exists():
                continue
            
            split_info = {
                'images': 0,
                'annotations': 0,
                'classes': defaultdict(int)
            }
            
            unique_images = set()
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    unique_images.add(row['filename'])
                    class_name = row['class']
                    split_info['classes'][class_name] += 1
                    split_info['annotations'] += 1
                    report['class_distribution'][class_name] += 1
            
            split_info['images'] = len(unique_images)
            report['splits'][split] = split_info
            report['total_images'] += split_info['images']
            report['total_annotations'] += split_info['annotations']
        
        return report
    
    def print_summary(self):
        """Imprime resumen del dataset"""
        report = self.create_summary_report()
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL DATASET DENTAL RADIOGRAPHY")
        print("="*60)
        
        print(f"📁 Dataset: {report['dataset_name']}")
        print(f"🖼️  Total de imágenes: {report['total_images']}")
        print(f"🏷️  Total de anotaciones: {report['total_annotations']}")
        
        print(f"\n📋 CLASES DISPONIBLES:")
        for class_name, class_id in report['classes'].items():
            count = report['class_distribution'][class_name]
            print(f"   • {class_name} (ID: {class_id}): {count} anotaciones")
        
        print(f"\n📊 DISTRIBUCIÓN POR SPLIT:")
        for split, info in report['splits'].items():
            print(f"   📁 {split.upper()}:")
            print(f"      🖼️  Imágenes: {info['images']}")
            print(f"      🏷️  Anotaciones: {info['annotations']}")
            for class_name, count in info['classes'].items():
                print(f"         • {class_name}: {count}")
        
        print("\n🎯 FORMATOS DISPONIBLES PARA CONVERSIÓN:")
        print("   • YOLO (txt + yaml)")
        print("   • COCO (json)")
        
def main():
    # Configuración
    base_path = Path(".")  # Ejecuta desde la carpeta Dental Radiography
    
    # Verificar que estamos en la carpeta correcta
    if not all((base_path / split).exists() for split in ['train', 'valid', 'test']):
        print("❌ Error: No se encuentran las carpetas 'train', 'valid', 'test'")
        print("   Ejecuta este script desde la carpeta 'Dental Radiography'")
        return
    
    converter = DentalAnnotationConverter(base_path)
    
    # Mostrar información del dataset
    converter.print_summary()
    
    # Preguntar qué formatos generar
    print("\n" + "="*60)
    print("🚀 CONVERSIÓN DE FORMATOS")
    print("="*60)
    
    while True:
        print("\n¿Qué formato deseas generar?")
        print("1. YOLO (txt + yaml)")
        print("2. COCO (json)")
        print("3. Ambos formatos")
        print("4. Solo mostrar resumen")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ").strip()
        
        if choice == '1':
            converter.csv_to_yolo()
        elif choice == '2':
            converter.csv_to_coco()
        elif choice == '3':
            converter.csv_to_yolo()
            converter.csv_to_coco()
        elif choice == '4':
            converter.print_summary()
        elif choice == '5':
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Intenta de nuevo.")
            continue
        
        if choice in ['1', '2', '3']:
            print(f"\n✅ ¡Conversión completada!")
            print("📁 Revisa las carpetas generadas:")
            if choice in ['1', '3']:
                print("   • dental_yolo/ (formato YOLO)")
            if choice in ['2', '3']:
                print("   • dental_coco/ (formato COCO)")

if __name__ == "__main__":
    main()
