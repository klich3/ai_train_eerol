#!/usr/bin/env python3
"""
🔍 DENTAL DATASET PREVIEW TOOL
==============================

Herramienta para previsualizar anotaciones en datasets dentales
Soporta formatos: YOLO, COCO, CSV y JSON
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple, Any

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

class DatasetPreviewTool:
    """🔍 Herramienta de previsualización de datasets."""
    
    def __init__(self):
        """Inicializar la herramienta."""
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080',
            '#008000', '#800000', '#808000', '#C0C0C0', '#FF6347', '#4682B4',
            '#D2691E', '#9ACD32', '#20B2AA', '#87CEEB', '#6495ED', '#DC143C'
        ]
        
        # Clases dentales comunes
        self.dental_classes = {
            'caries': 'Caries',
            'tooth': 'Diente',
            'filling': 'Empaste',
            'crown': 'Corona',
            'implant': 'Implante',
            'root_canal': 'Endodoncia',
            'bone_loss': 'Pérdida Ósea',
            'impacted': 'Impactado',
            'periapical_lesion': 'Lesión Periapical',
            'maxillary_sinus': 'Seno Maxilar',
            'mandible': 'Mandíbula',
            'maxilla': 'Maxilar'
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Cargar imagen desde archivo."""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        # Cargar imagen con OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_yolo_annotations(self, txt_path: str, img_shape: Tuple[int, int]) -> List[Dict]:
        """Cargar anotaciones YOLO desde archivo .txt"""
        annotations = []
        txt_file = Path(txt_path)
        
        if not txt_file.exists():
            print(f"⚠️ Archivo de anotaciones no encontrado: {txt_path}")
            return annotations
        
        height, width = img_shape[:2]
        
        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        print(f"⚠️ Línea {line_num} inválida: {line}")
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])
                    
                    # Convertir coordenadas normalizadas a píxeles
                    x_center_px = x_center * width
                    y_center_px = y_center * height
                    bbox_width_px = bbox_width * width
                    bbox_height_px = bbox_height * height
                    
                    # Calcular esquina superior izquierda
                    x1 = x_center_px - bbox_width_px / 2
                    y1 = y_center_px - bbox_height_px / 2
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, bbox_width_px, bbox_height_px],
                        'format': 'yolo'
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error en línea {line_num}: {e}")
        
        return annotations
    
    def load_coco_annotations(self, json_path: str, image_filename: str) -> List[Dict]:
        """Cargar anotaciones COCO desde archivo .json"""
        annotations = []
        json_file = Path(json_path)
        
        if not json_file.exists():
            print(f"⚠️ Archivo de anotaciones no encontrado: {json_path}")
            return annotations
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Encontrar la imagen
            image_id = None
            for img in coco_data.get('images', []):
                if img['file_name'] == image_filename:
                    image_id = img['id']
                    break
            
            if image_id is None:
                print(f"⚠️ Imagen {image_filename} no encontrada en COCO")
                return annotations
            
            # Crear mapeo de categorías
            categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            
            # Obtener anotaciones para esta imagen
            for ann in coco_data.get('annotations', []):
                if ann['image_id'] == image_id:
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    annotation = {
                        'class_id': ann['category_id'],
                        'class_name': categories.get(ann['category_id'], f"Class_{ann['category_id']}"),
                        'bbox': bbox,
                        'format': 'coco'
                    }
                    
                    # Agregar segmentación si existe
                    if 'segmentation' in ann:
                        annotation['segmentation'] = ann['segmentation']
                    
                    annotations.append(annotation)
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error al cargar COCO: {e}")
        
        return annotations
    
    def load_csv_annotations(self, csv_path: str, image_filename: str) -> List[Dict]:
        """Cargar anotaciones desde archivo CSV."""
        annotations = []
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            print(f"⚠️ Archivo CSV no encontrado: {csv_path}")
            return annotations
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # Filtrar por nombre de imagen
            img_rows = df[df['filename'] == image_filename]
            
            for _, row in img_rows.iterrows():
                annotation = {
                    'class_name': row.get('class', 'Unknown'),
                    'bbox': [
                        row.get('xmin', 0),
                        row.get('ymin', 0),
                        row.get('xmax', 0) - row.get('xmin', 0),  # width
                        row.get('ymax', 0) - row.get('ymin', 0)   # height
                    ],
                    'format': 'csv'
                }
                annotations.append(annotation)
        
        except Exception as e:
            print(f"⚠️ Error al cargar CSV: {e}")
        
        return annotations
    
    def load_json_annotations(self, json_path: str, image_filename: str) -> List[Dict]:
        """Cargar anotaciones desde archivo JSON personalizado."""
        annotations = []
        json_file = Path(json_path)
        
        if not json_file.exists():
            print(f"⚠️ Archivo JSON no encontrado: {json_path}")
            return annotations
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Buscar anotaciones para esta imagen
            if image_filename in data:
                img_annotations = data[image_filename]
                for ann in img_annotations:
                    annotations.append({
                        'class_name': ann.get('class', 'Unknown'),
                        'bbox': ann.get('bbox', [0, 0, 0, 0]),
                        'format': 'json'
                    })
        
        except Exception as e:
            print(f"⚠️ Error al cargar JSON: {e}")
        
        return annotations
    
    def draw_annotations(self, image: np.ndarray, annotations: List[Dict], 
                        classes_file: str = None) -> np.ndarray:
        """Dibujar anotaciones sobre la imagen."""
        img_with_annotations = image.copy()
        
        # Cargar nombres de clases si se proporciona archivo
        class_names = {}
        if classes_file and Path(classes_file).exists():
            with open(classes_file, 'r') as f:
                class_names = {i: line.strip() for i, line in enumerate(f)}
        
        for i, ann in enumerate(annotations):
            # Obtener color
            color = self.colors[i % len(self.colors)]
            color_rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
            
            # Obtener bbox
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Dibujar rectángulo
            cv2.rectangle(img_with_annotations, 
                         (int(x), int(y)), 
                         (int(x + w), int(y + h)), 
                         color_rgb, 2)
            
            # Obtener nombre de clase
            if 'class_name' in ann:
                class_name = ann['class_name']
            elif 'class_id' in ann and ann['class_id'] in class_names:
                class_name = class_names[ann['class_id']]
            elif 'class_id' in ann:
                class_name = f"Class_{ann['class_id']}"
            else:
                class_name = "Unknown"
            
            # Dibujar etiqueta
            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fondo para el texto
            cv2.rectangle(img_with_annotations,
                         (int(x), int(y) - label_size[1] - 10),
                         (int(x) + label_size[0], int(y)),
                         color_rgb, -1)
            
            # Texto
            cv2.putText(img_with_annotations, label,
                       (int(x), int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img_with_annotations
    
    def preview_dataset(self, image_path: str, annotation_path: str, 
                       format_type: str = 'auto', classes_file: str = None,
                       save_output: str = None):
        """Previsualizar dataset con anotaciones."""
        
        print(f"🔍 PREVISUALIZANDO DATASET")
        print(f"="*40)
        print(f"📸 Imagen: {image_path}")
        print(f"📋 Anotaciones: {annotation_path}")
        print(f"📊 Formato: {format_type}")
        
        # Cargar imagen
        try:
            image = self.load_image(image_path)
            print(f"✅ Imagen cargada: {image.shape}")
        except Exception as e:
            print(f"❌ Error al cargar imagen: {e}")
            return
        
        # Detectar formato automáticamente si es necesario
        if format_type == 'auto':
            ext = Path(annotation_path).suffix.lower()
            if ext == '.txt':
                format_type = 'yolo'
            elif ext == '.json':
                format_type = 'coco'  # Asumimos COCO por defecto
            elif ext == '.csv':
                format_type = 'csv'
            else:
                print(f"⚠️ No se pudo detectar el formato. Usando YOLO por defecto.")
                format_type = 'yolo'
        
        # Cargar anotaciones según el formato
        annotations = []
        image_filename = Path(image_path).name
        
        if format_type == 'yolo':
            annotations = self.load_yolo_annotations(annotation_path, image.shape)
        elif format_type == 'coco':
            annotations = self.load_coco_annotations(annotation_path, image_filename)
        elif format_type == 'csv':
            annotations = self.load_csv_annotations(annotation_path, image_filename)
        elif format_type == 'json':
            annotations = self.load_json_annotations(annotation_path, image_filename)
        
        print(f"📊 Anotaciones encontradas: {len(annotations)}")
        
        # Dibujar anotaciones
        if annotations:
            annotated_image = self.draw_annotations(image, annotations, classes_file)
            
            # Mostrar estadísticas
            print(f"\n📈 ESTADÍSTICAS:")
            format_counts = {}
            for ann in annotations:
                class_name = ann.get('class_name', f"Class_{ann.get('class_id', 'Unknown')}")
                format_counts[class_name] = format_counts.get(class_name, 0) + 1
            
            for class_name, count in format_counts.items():
                print(f"   • {class_name}: {count}")
        else:
            annotated_image = image
            print("⚠️ No se encontraron anotaciones")
        
        # Mostrar imagen
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Imagen Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_image)
        plt.title(f'Con Anotaciones ({len(annotations)} objetos)')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if save_output:
            plt.savefig(save_output, dpi=300, bbox_inches='tight')
            print(f"💾 Guardado en: {save_output}")
        
        plt.show()
        
        return annotated_image, annotations

def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="🔍 Herramienta de previsualización de datasets dentales"
    )
    
    parser.add_argument("image", help="Ruta a la imagen")
    parser.add_argument("annotations", help="Ruta al archivo de anotaciones")
    parser.add_argument("--format", "-f", choices=['yolo', 'coco', 'csv', 'json', 'auto'],
                       default='auto', help="Formato de anotaciones")
    parser.add_argument("--classes", "-c", help="Archivo de clases (opcional)")
    parser.add_argument("--output", "-o", help="Guardar imagen anotada")
    
    args = parser.parse_args()
    
    # Crear herramienta
    tool = DatasetPreviewTool()
    
    # Previsualizar
    try:
        tool.preview_dataset(
            image_path=args.image,
            annotation_path=args.annotations,
            format_type=args.format,
            classes_file=args.classes,
            save_output=args.output
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
