#!/usr/bin/env python3
"""
🧪 Demo de Resultados
Simula resultados de GPU para demostrar el procesamiento
"""

import json
import random
from pathlib import Path
import cv2
import numpy as np

def create_demo_results(batch_dir, output_dir):
    """Crea resultados demo para mostrar el procesamiento."""
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎭 Creando resultados demo para: {batch_dir.name}")
    
    # Leer info del lote
    batch_info_file = batch_dir / "batch_info.json"
    if batch_info_file.exists():
        with open(batch_info_file, 'r') as f:
            batch_info = json.load(f)
    else:
        batch_info = {"batch_id": batch_dir.name}
    
    # Clases dentales de ejemplo
    dental_classes = ["tooth", "caries", "filling", "crown", "implant", "other"]
    
    # Buscar imágenes en el lote
    image_files = []
    for split_dir in batch_dir.glob("*/"):
        if split_dir.is_dir() and split_dir.name not in ["__pycache__"]:
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            image_files.extend(images)
    
    if not image_files:
        print("❌ No se encontraron imágenes en el lote")
        return
    
    # Crear resultados simulados
    results_summary = {
        'batch_info': batch_info,
        'total_images': len(image_files),
        'results': []
    }
    
    print(f"🖼️ Simulando resultados para {len(image_files)} imágenes...")
    
    for img_file in image_files:
        # Simular detecciones (cantidad aleatoria)
        num_detections = random.randint(0, 4)
        detections = []
        
        for i in range(num_detections):
            # Coordenadas aleatorias (formato xyxy)
            x1, y1 = random.randint(50, 300), random.randint(50, 300)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            
            detection = {
                'class_id': random.randint(0, len(dental_classes)-1),
                'class': random.choice(dental_classes),
                'confidence': round(random.uniform(0.3, 0.95), 3),
                'bbox': [x1, y1, x2, y2]
            }
            detections.append(detection)
        
        results_summary['results'].append({
            'image': img_file.name,
            'detections': detections
        })
        
        print(f"  📸 {img_file.name}: {num_detections} detecciones")
    
    # Guardar resultados
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Crear imágenes con detecciones marcadas
    print(f"🎨 Creando imágenes marcadas...")
    
    for result in results_summary['results']:
        img_name = result['image']
        detections = result['detections']
        
        # Buscar la imagen original
        original_img_path = None
        for img_file in image_files:
            if img_file.name == img_name:
                original_img_path = img_file
                break
        
        if not original_img_path or not original_img_path.exists():
            print(f"⚠️ No se encontró imagen original: {img_name}")
            continue
            
        try:
            # Cargar imagen original
            img = cv2.imread(str(original_img_path))
            if img is None:
                print(f"⚠️ No se pudo cargar imagen: {img_name}")
                continue
                
            # Obtener dimensiones
            height, width = img.shape[:2]
            
            # Dibujar detecciones
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Coordenadas del bounding box
                x1, y1, x2, y2 = bbox
                
                # Asegurar que las coordenadas estén dentro de la imagen
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(x1+1, min(x2, width))
                y2 = max(y1+1, min(y2, height))
                
                # Color según la clase (diferentes colores para diferentes clases)
                colors = {
                    'tooth': (0, 255, 0),      # Verde
                    'caries': (0, 0, 255),     # Rojo
                    'filling': (255, 0, 0),    # Azul
                    'crown': (255, 255, 0),    # Cian
                    'implant': (255, 0, 255),  # Magenta
                    'other': (128, 128, 128)   # Gris
                }
                color = colors.get(class_name, (255, 255, 255))
                
                # Dibujar rectángulo
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Preparar texto
                label = f"{class_name}: {confidence:.2f}"
                
                # Calcular tamaño del texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Fondo para el texto
                cv2.rectangle(img, 
                            (int(x1), int(y1) - text_height - baseline - 5),
                            (int(x1) + text_width, int(y1)),
                            color, -1)
                
                # Texto
                cv2.putText(img, label, 
                          (int(x1), int(y1) - baseline - 2),
                          font, font_scale, (0, 0, 0), thickness)
            
            # Guardar imagen marcada
            output_img_path = output_dir / f"predicted_{img_name}"
            cv2.imwrite(str(output_img_path), img)
            print(f"  ✅ {img_name} -> {output_img_path.name}")
            
        except Exception as e:
            print(f"❌ Error procesando {img_name}: {e}")
    
    print(f"✅ Resultados demo creados en: {output_dir}")
    print(f"📄 Archivo principal: {results_file}")
    print(f"🖼️ Imágenes marcadas: {len([f for f in output_dir.glob('predicted_*') if f.is_file()])}")
    
    return output_dir

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Uso: python create_demo_results.py <batch_dir> <output_dir>")
        sys.exit(1)
    
    batch_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    create_demo_results(batch_dir, output_dir)

if __name__ == "__main__":
    main()
