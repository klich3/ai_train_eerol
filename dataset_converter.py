#!/usr/bin/env python3
"""
🔄 DATASET FORMAT CONVERTER
===========================

Herramienta para convertir datasets entre diferentes formatos:
YOLO ↔ COCO ↔ U-Net ↔ Clasificación

Conversiones soportadas:
- YOLO → COCO
- COCO → YOLO
- Clasificación → YOLO
- U-Net → YOLO
- YOLO → U-Net
- Cualquier formato → Clasificación
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml


class FormatConverter:
    """🔄 Convertidor de formatos de datasets."""
    
    def __init__(self):
        self.supported_conversions = {
            'yolo_to_coco': self._yolo_to_coco,
            'coco_to_yolo': self._coco_to_yolo,
            'classification_to_yolo': self._classification_to_yolo,
            'unet_to_yolo': self._unet_to_yolo,
            'yolo_to_unet': self._yolo_to_unet,
            'any_to_classification': self._any_to_classification
        }
    
    def convert_dataset(self, source_path: str, target_path: str, 
                       source_format: str, target_format: str,
                       class_names: Optional[List[str]] = None) -> bool:
        """
        🔄 Convierte un dataset de un formato a otro.
        
        Args:
            source_path: Ruta del dataset origen
            target_path: Ruta del dataset destino
            source_format: Formato origen (yolo, coco, unet, classification)
            target_format: Formato destino
            class_names: Lista de nombres de clases (opcional)
        """
        conversion_key = f"{source_format}_to_{target_format}"
        
        if conversion_key not in self.supported_conversions:
            print(f"❌ Conversión {source_format} → {target_format} no soportada")
            return False
        
        try:
            print(f"🔄 Convirtiendo {source_format.upper()} → {target_format.upper()}")
            print(f"📂 Origen: {source_path}")
            print(f"📁 Destino: {target_path}")
            
            # Crear directorio destino
            Path(target_path).mkdir(parents=True, exist_ok=True)
            
            # Ejecutar conversión
            converter_func = self.supported_conversions[conversion_key]
            result = converter_func(source_path, target_path, class_names)
            
            if result:
                print(f"✅ Conversión completada exitosamente")
            else:
                print(f"❌ Error en la conversión")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en conversión: {e}")
            return False
    
    def _yolo_to_coco(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """🎯→🎨 Convierte YOLO a COCO."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Buscar clases si no se proporcionan
        if not class_names:
            class_names = self._extract_yolo_classes(source)
        
        # Estructura COCO
        coco_data = {
            "info": {
                "description": f"Converted from YOLO dataset at {datetime.now()}",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Dataset Organizer",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Crear categorías
        for i, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "dental"
            })
        
        # Procesar imágenes y anotaciones
        image_id = 0
        annotation_id = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = source / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists():
                continue
            
            # Crear directorio de salida para este split
            target_split = target / split
            target_split.mkdir(exist_ok=True)
            
            for img_file in images_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copiar imagen
                    shutil.copy2(img_file, target_split / img_file.name)
                    
                    # Leer imagen para obtener dimensiones
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # Agregar imagen a COCO
                    coco_data["images"].append({
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": img_file.name,
                        "license": 0,
                        "flickr_url": "",
                        "coco_url": "",
                        "date_captured": 0
                    })
                    
                    # Procesar anotaciones YOLO
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    box_width = float(parts[3])
                                    box_height = float(parts[4])
                                    
                                    # Convertir de YOLO a COCO bbox
                                    x = (x_center - box_width/2) * width
                                    y = (y_center - box_height/2) * height
                                    w = box_width * width
                                    h = box_height * height
                                    
                                    coco_data["annotations"].append({
                                        "id": annotation_id,
                                        "image_id": image_id,
                                        "category_id": class_id,
                                        "bbox": [x, y, w, h],
                                        "area": w * h,
                                        "iscrowd": 0
                                    })
                                    
                                    annotation_id += 1
                    
                    image_id += 1
            
            # Guardar archivo de anotaciones para este split
            with open(target_split / f"_annotations.coco.json", 'w') as f:
                json.dump(coco_data, f, indent=2)
        
        return True
    
    def _coco_to_yolo(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """🎨→🎯 Convierte COCO a YOLO."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Buscar archivos de anotaciones COCO
        annotation_files = list(source.glob('**/*.json'))
        
        if not annotation_files:
            print("❌ No se encontraron archivos de anotaciones COCO")
            return False
        
        # Procesar cada archivo de anotaciones
        all_classes = set()
        
        for ann_file in annotation_files:
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Extraer nombres de clases
            categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            all_classes.update(categories.values())
            
            # Determinar split basado en el nombre del archivo
            split = 'train'
            if 'val' in ann_file.name.lower():
                split = 'val'
            elif 'test' in ann_file.name.lower():
                split = 'test'
            
            # Crear directorios YOLO
            yolo_images = target / split / 'images'
            yolo_labels = target / split / 'labels'
            yolo_images.mkdir(parents=True, exist_ok=True)
            yolo_labels.mkdir(parents=True, exist_ok=True)
            
            # Procesar imágenes y anotaciones
            images = {img['id']: img for img in coco_data.get('images', [])}
            
            for image_id, image_info in images.items():
                # Copiar imagen
                src_img = source / image_info['file_name']
                if src_img.exists():
                    shutil.copy2(src_img, yolo_images / image_info['file_name'])
                
                # Crear archivo de etiquetas YOLO
                img_width = image_info['width']
                img_height = image_info['height']
                
                label_file = yolo_labels / f"{Path(image_info['file_name']).stem}.txt"
                
                with open(label_file, 'w') as f:
                    for ann in coco_data.get('annotations', []):
                        if ann['image_id'] == image_id:
                            # Convertir bbox de COCO a YOLO
                            x, y, w, h = ann['bbox']
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height
                            
                            class_id = ann['category_id']
                            f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
        
        # Crear archivo data.yaml para YOLO
        class_list = sorted(list(all_classes))
        data_yaml = {
            'train': str(target / 'train' / 'images'),
            'val': str(target / 'val' / 'images'),
            'test': str(target / 'test' / 'images'),
            'nc': len(class_list),
            'names': class_list
        }
        
        with open(target / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return True
    
    def _classification_to_yolo(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """📁→🎯 Convierte clasificación a YOLO (imágenes completas como una clase)."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Encontrar carpetas de clases
        class_dirs = [d for d in source.iterdir() if d.is_dir()]
        
        if not class_dirs:
            print("❌ No se encontraron carpetas de clases")
            return False
        
        # Crear estructura YOLO
        for split in ['train', 'val']:
            (target / split / 'images').mkdir(parents=True, exist_ok=True)
            (target / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        class_names = [d.name for d in class_dirs]
        image_count = 0
        
        for class_id, class_dir in enumerate(class_dirs):
            class_images = list(class_dir.glob('*'))
            class_images = [f for f in class_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            # Dividir en train/val (80/20)
            split_point = int(len(class_images) * 0.8)
            train_images = class_images[:split_point]
            val_images = class_images[split_point:]
            
            for split, images in [('train', train_images), ('val', val_images)]:
                for img_file in images:
                    # Copiar imagen
                    new_name = f"{class_dir.name}_{img_file.name}"
                    shutil.copy2(img_file, target / split / 'images' / new_name)
                    
                    # Crear etiqueta YOLO (toda la imagen)
                    label_file = target / split / 'labels' / f"{Path(new_name).stem}.txt"
                    with open(label_file, 'w') as f:
                        # Toda la imagen como una bounding box
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    
                    image_count += 1
        
        # Crear data.yaml
        data_yaml = {
            'train': str(target / 'train' / 'images'),
            'val': str(target / 'val' / 'images'),
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(target / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ Procesadas {image_count} imágenes en {len(class_names)} clases")
        return True
    
    def _unet_to_yolo(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """🎭→🎯 Convierte U-Net (máscaras) a YOLO."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Buscar carpetas de imágenes y máscaras
        images_dir = None
        masks_dir = None
        
        for item in source.iterdir():
            if item.is_dir():
                if 'image' in item.name.lower():
                    images_dir = item
                elif 'mask' in item.name.lower() or 'label' in item.name.lower():
                    masks_dir = item
        
        if not images_dir or not masks_dir:
            print("❌ No se encontraron carpetas de imágenes y máscaras")
            return False
        
        # Crear estructura YOLO
        for split in ['train', 'val']:
            (target / split / 'images').mkdir(parents=True, exist_ok=True)
            (target / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Procesar imágenes y máscaras
        image_files = list(images_dir.glob('*'))
        processed = 0
        
        for img_file in image_files:
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Buscar máscara correspondiente
            mask_file = masks_dir / img_file.name
            if not mask_file.exists():
                # Probar con extensión PNG
                mask_file = masks_dir / f"{img_file.stem}.png"
            
            if not mask_file.exists():
                continue
            
            # Leer imagen y máscara
            image = cv2.imread(str(img_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                continue
            
            height, width = image.shape[:2]
            
            # Determinar split (80% train, 20% val)
            split = 'train' if processed % 5 != 0 else 'val'
            
            # Copiar imagen
            shutil.copy2(img_file, target / split / 'images' / img_file.name)
            
            # Convertir máscara a bounding boxes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            label_file = target / split / 'labels' / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                for contour in contours:
                    if cv2.contourArea(contour) < 100:  # Filtrar contornos muy pequeños
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Normalizar coordinates
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    norm_width = w / width
                    norm_height = h / height
                    
                    f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")
            
            processed += 1
        
        # Crear data.yaml
        data_yaml = {
            'train': str(target / 'train' / 'images'),
            'val': str(target / 'val' / 'images'),
            'nc': 1,
            'names': ['object']
        }
        
        with open(target / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✅ Procesadas {processed} imágenes con máscaras")
        return True
    
    def _yolo_to_unet(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """🎯→🎭 Convierte YOLO a U-Net (crear máscaras desde bounding boxes)."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Crear estructura U-Net
        (target / 'images').mkdir(parents=True, exist_ok=True)
        (target / 'masks').mkdir(parents=True, exist_ok=True)
        
        processed = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = source / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists():
                continue
            
            for img_file in images_dir.glob('*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Copiar imagen
                shutil.copy2(img_file, target / 'images' / img_file.name)
                
                # Leer imagen para obtener dimensiones
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                
                # Crear máscara
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Leer anotaciones YOLO
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                box_width = float(parts[3])
                                box_height = float(parts[4])
                                
                                # Convertir a coordenadas de píxeles
                                x = int((x_center - box_width/2) * width)
                                y = int((y_center - box_height/2) * height)
                                w = int(box_width * width)
                                h = int(box_height * height)
                                
                                # Dibujar rectángulo en la máscara
                                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                
                # Guardar máscara
                cv2.imwrite(str(target / 'masks' / f"{img_file.stem}.png"), mask)
                processed += 1
        
        print(f"✅ Procesadas {processed} imágenes con máscaras generadas")
        return True
    
    def _any_to_classification(self, source_path: str, target_path: str, class_names: List[str]) -> bool:
        """📄→📁 Convierte cualquier formato a clasificación por carpetas."""
        source = Path(source_path)
        target = Path(target_path)
        
        # Esta conversión es más compleja y requiere análisis del contenido
        # Por ahora, implementamos una versión básica
        
        if not class_names:
            class_names = ['class_0', 'class_1']
        
        # Crear carpetas de clases
        for class_name in class_names:
            (target / class_name).mkdir(parents=True, exist_ok=True)
        
        # Buscar todas las imágenes
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(source.rglob(ext))
        
        # Distribuir imágenes por clases (distribución simple)
        images_per_class = len(image_files) // len(class_names)
        
        for i, img_file in enumerate(image_files):
            class_idx = min(i // max(1, images_per_class), len(class_names) - 1)
            class_name = class_names[class_idx]
            
            shutil.copy2(img_file, target / class_name / img_file.name)
        
        print(f"✅ Distribuidas {len(image_files)} imágenes en {len(class_names)} clases")
        return True
    
    def _extract_yolo_classes(self, yolo_path: Path) -> List[str]:
        """📋 Extrae nombres de clases de un dataset YOLO."""
        # Buscar archivo data.yaml o classes.txt
        data_yaml = yolo_path / 'data.yaml'
        classes_txt = yolo_path / 'classes.txt'
        
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        
        elif classes_txt.exists():
            with open(classes_txt, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        
        else:
            # Generar nombres por defecto
            return [f'class_{i}' for i in range(10)]


def main():
    """🚀 Función principal del convertidor."""
    print("🔄 DATASET FORMAT CONVERTER")
    print("="*35)
    print()
    
    converter = FormatConverter()
    
    print("🎯 CONVERSIONES DISPONIBLES:")
    print("1. YOLO → COCO")
    print("2. COCO → YOLO")
    print("3. Clasificación → YOLO")
    print("4. U-Net → YOLO")
    print("5. YOLO → U-Net")
    print("6. Cualquier → Clasificación")
    print("0. Salir")
    
    choice = input("\n🔄 Selecciona conversión: ").strip()
    
    conversions = {
        '1': ('yolo', 'coco'),
        '2': ('coco', 'yolo'),
        '3': ('classification', 'yolo'),
        '4': ('unet', 'yolo'),
        '5': ('yolo', 'unet'),
        '6': ('any', 'classification')
    }
    
    if choice not in conversions:
        print("👋 ¡Hasta luego!")
        return
    
    source_format, target_format = conversions[choice]
    
    # Solicitar rutas
    source_path = input("📂 Ruta del dataset origen: ").strip()
    target_path = input("📁 Ruta del dataset destino: ").strip()
    
    if not Path(source_path).exists():
        print("❌ La ruta origen no existe")
        return
    
    # Solicitar nombres de clases (opcional)
    class_names_input = input("📋 Nombres de clases (separados por coma, opcional): ").strip()
    class_names = [name.strip() for name in class_names_input.split(',')] if class_names_input else None
    
    # Ejecutar conversión
    success = converter.convert_dataset(
        source_path, target_path, 
        source_format, target_format, 
        class_names
    )
    
    if success:
        print(f"\n🎉 ¡Conversión completada exitosamente!")
        print(f"📁 Dataset convertido disponible en: {target_path}")
    else:
        print(f"\n❌ Error en la conversión")


if __name__ == "__main__":
    main()
