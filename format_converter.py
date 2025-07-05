#!/usr/bin/env python3
"""
🔄 Convertidor de Formatos de Datasets Dentales
===============================================

Convierte datasets entre diferentes formatos:
YOLO ↔ COCO ↔ Pascal VOC ↔ U-Net ↔ Clasificación

Author: Anton Sychev
Created: 2025-07-05
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))


class DatasetFormatConverter:
    """🔄 Convertidor de formatos de datasets"""
    
    def __init__(self):
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.conversion_stats = {
            'images_processed': 0,
            'annotations_converted': 0,
            'errors': 0
        }
    
    def yolo_to_coco(self, yolo_path: Path, output_path: Path, dataset_name: str = "dental_dataset") -> bool:
        """🔄 Convierte YOLO a formato COCO"""
        print(f"🔄 Convirtiendo YOLO → COCO")
        print(f"   📂 Origen: {yolo_path}")
        print(f"   📁 Destino: {output_path}")
        
        try:
            # Crear estructura COCO
            output_path.mkdir(parents=True, exist_ok=True)
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Buscar imágenes y anotaciones YOLO
            image_files = []
            for ext in self.supported_image_extensions:
                image_files.extend(yolo_path.glob(f"**/*{ext}"))
            
            if not image_files:
                print("❌ No se encontraron imágenes")
                return False
            
            # Estructura COCO
            coco_data = {
                "info": {
                    "description": f"Converted from YOLO - {dataset_name}",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "Auto-converter",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "dental_object", "supercategory": "medical"}]
            }
            
            annotation_id = 1
            
            for image_id, image_file in enumerate(image_files, 1):
                # Copiar imagen
                dest_image = images_dir / image_file.name
                if not dest_image.exists():
                    import shutil
                    shutil.copy2(image_file, dest_image)
                
                # Leer dimensiones de imagen
                img = cv2.imread(str(image_file))
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                
                # Agregar info de imagen
                coco_data["images"].append({
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": image_file.name,
                    "license": 1,
                    "date_captured": ""
                })
                
                # Buscar archivo de anotaciones YOLO
                txt_file = yolo_path / f"{image_file.stem}.txt"
                if not txt_file.exists():
                    # Buscar en subcarpetas
                    possible_txt = list(yolo_path.rglob(f"{image_file.stem}.txt"))
                    if possible_txt:
                        txt_file = possible_txt[0]
                    else:
                        continue
                
                # Convertir anotaciones YOLO a COCO
                try:
                    with open(txt_file, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1]) * width
                                y_center = float(parts[2]) * height
                                bbox_width = float(parts[3]) * width
                                bbox_height = float(parts[4]) * height
                                
                                # Convertir a formato COCO (x, y, width, height)
                                x = x_center - bbox_width / 2
                                y = y_center - bbox_height / 2
                                
                                area = bbox_width * bbox_height
                                
                                coco_data["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1,  # Dental object
                                    "bbox": [x, y, bbox_width, bbox_height],
                                    "area": area,
                                    "segmentation": [],
                                    "iscrowd": 0
                                })
                                
                                annotation_id += 1
                                self.conversion_stats['annotations_converted'] += 1
                
                except Exception as e:
                    print(f"⚠️ Error procesando {txt_file.name}: {e}")
                    self.conversion_stats['errors'] += 1
                
                self.conversion_stats['images_processed'] += 1
            
            # Guardar archivo COCO JSON
            with open(output_path / "annotations.json", 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"✅ Conversión completada:")
            print(f"   📊 Imágenes: {self.conversion_stats['images_processed']}")
            print(f"   🏷️ Anotaciones: {self.conversion_stats['annotations_converted']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en conversión YOLO→COCO: {e}")
            return False
    
    def coco_to_yolo(self, coco_path: Path, output_path: Path) -> bool:
        """🔄 Convierte COCO a formato YOLO"""
        print(f"🔄 Convirtiendo COCO → YOLO")
        print(f"   📂 Origen: {coco_path}")
        print(f"   📁 Destino: {output_path}")
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Buscar archivo de anotaciones COCO
            json_files = list(coco_path.glob("*.json"))
            if not json_files:
                json_files = list(coco_path.glob("**/*.json"))
            
            if not json_files:
                print("❌ No se encontró archivo de anotaciones COCO")
                return False
            
            coco_file = json_files[0]
            
            # Cargar datos COCO
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Crear mapeo de imágenes
            images_map = {img['id']: img for img in coco_data['images']}
            
            # Procesar anotaciones
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # Convertir cada imagen
            for image_id, image_info in images_map.items():
                # Buscar y copiar imagen
                image_name = image_info['file_name']
                
                # Buscar imagen en el directorio COCO
                image_paths = list(coco_path.glob(f"**/{image_name}"))
                if not image_paths:
                    print(f"⚠️ No se encontró imagen: {image_name}")
                    continue
                
                source_image = image_paths[0]
                dest_image = output_path / image_name
                
                if not dest_image.exists():
                    import shutil
                    shutil.copy2(source_image, dest_image)
                
                # Convertir anotaciones a YOLO
                yolo_annotations = []
                
                if image_id in annotations_by_image:
                    width = image_info['width']
                    height = image_info['height']
                    
                    for ann in annotations_by_image[image_id]:
                        bbox = ann['bbox']  # [x, y, width, height]
                        
                        # Convertir a formato YOLO
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        bbox_width = bbox[2] / width
                        bbox_height = bbox[3] / height
                        
                        # Usar category_id - 1 como class_id (YOLO usa índices desde 0)
                        class_id = ann['category_id'] - 1
                        
                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
                        self.conversion_stats['annotations_converted'] += 1
                
                # Guardar archivo YOLO
                txt_file = output_path / f"{Path(image_name).stem}.txt"
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                self.conversion_stats['images_processed'] += 1
            
            print(f"✅ Conversión completada:")
            print(f"   📊 Imágenes: {self.conversion_stats['images_processed']}")
            print(f"   🏷️ Anotaciones: {self.conversion_stats['annotations_converted']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en conversión COCO→YOLO: {e}")
            return False
    
    def pascal_voc_to_yolo(self, voc_path: Path, output_path: Path) -> bool:
        """🔄 Convierte Pascal VOC a formato YOLO"""
        print(f"🔄 Convirtiendo Pascal VOC → YOLO")
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Buscar archivos XML
            xml_files = list(voc_path.glob("**/*.xml"))
            if not xml_files:
                print("❌ No se encontraron archivos XML")
                return False
            
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # Información de la imagen
                    filename = root.find('filename').text
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    
                    # Buscar imagen correspondiente
                    image_paths = []
                    for ext in self.supported_image_extensions:
                        image_paths.extend(voc_path.glob(f"**/{Path(filename).stem}{ext}"))
                        image_paths.extend(voc_path.glob(f"**/{filename}"))
                    
                    if image_paths:
                        # Copiar imagen
                        source_image = image_paths[0]
                        dest_image = output_path / source_image.name
                        if not dest_image.exists():
                            import shutil
                            shutil.copy2(source_image, dest_image)
                    
                    # Convertir anotaciones
                    yolo_annotations = []
                    for obj in root.findall('object'):
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        # Convertir a formato YOLO
                        x_center = (xmin + xmax) / 2.0 / width
                        y_center = (ymin + ymax) / 2.0 / height
                        bbox_width = (xmax - xmin) / width
                        bbox_height = (ymax - ymin) / height
                        
                        yolo_annotations.append(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")
                        self.conversion_stats['annotations_converted'] += 1
                    
                    # Guardar archivo YOLO
                    txt_file = output_path / f"{Path(filename).stem}.txt"
                    with open(txt_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    self.conversion_stats['images_processed'] += 1
                    
                except Exception as e:
                    print(f"⚠️ Error procesando {xml_file.name}: {e}")
                    self.conversion_stats['errors'] += 1
            
            print(f"✅ Conversión completada:")
            print(f"   📊 Imágenes: {self.conversion_stats['images_processed']}")
            print(f"   🏷️ Anotaciones: {self.conversion_stats['annotations_converted']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en conversión Pascal VOC→YOLO: {e}")
            return False
    
    def run_interactive_converter(self):
        """🎮 Ejecuta el convertidor en modo interactivo"""
        print("🔄 CONVERTIDOR DE FORMATOS DE DATASETS")
        print("="*45)
        print()
        print("Formatos soportados:")
        print("1. YOLO → COCO")
        print("2. COCO → YOLO") 
        print("3. Pascal VOC → YOLO")
        print("4. Detectar formato automáticamente y convertir")
        print()
        
        choice = input("Selecciona una opción (1-4): ").strip()
        
        if choice not in ['1', '2', '3', '4']:
            print("❌ Opción inválida")
            return
        
        # Solicitar rutas
        source_path = input("📂 Ruta del dataset origen: ").strip()
        if not source_path:
            print("❌ Ruta de origen requerida")
            return
        
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"❌ No existe: {source_path}")
            return
        
        output_path = input("📁 Ruta de salida: ").strip()
        if not output_path:
            output_path = source_path.parent / f"{source_path.name}_converted"
        
        output_path = Path(output_path)
        
        # Reset stats
        self.conversion_stats = {k: 0 for k in self.conversion_stats.keys()}
        
        # Ejecutar conversión
        success = False
        
        if choice == '1':
            success = self.yolo_to_coco(source_path, output_path)
        elif choice == '2':
            success = self.coco_to_yolo(source_path, output_path)
        elif choice == '3':
            success = self.pascal_voc_to_yolo(source_path, output_path)
        elif choice == '4':
            # Auto-detectar formato
            print("🔍 Detectando formato automáticamente...")
            # Aquí podrías integrar con el detector del auto_organize_datasets.py
            print("🚧 Función en desarrollo")
        
        if success:
            print(f"\n🎉 ¡Conversión exitosa!")
            print(f"📁 Resultado guardado en: {output_path}")
        else:
            print(f"\n💥 Error en la conversión")


def main():
    """🚀 Función principal"""
    converter = DatasetFormatConverter()
    converter.run_interactive_converter()


if __name__ == "__main__":
    main()
