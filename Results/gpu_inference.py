#!/usr/bin/env python3
"""
🚀 GPU Inference Script
Script para ejecutar en máquina con GPU
"""

import sys
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

def run_inference(model_path, batch_dir, output_dir):
    """Ejecuta inferencia en un lote de imágenes."""
    model = YOLO(model_path)
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Leer manifiesto
    manifest_file = batch_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        print(f"📦 Procesando lote: {manifest['batch_id']}")
    
    # Buscar imágenes
    image_files = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.png"))
    
    results_summary = {
        'batch_info': manifest if manifest_file.exists() else {},
        'total_images': len(image_files),
        'results': []
    }
    
    for i, img_path in enumerate(image_files):
        print(f"🖼️ Procesando {i+1}/{len(image_files)}: {img_path.name}")
        
        # Inferencia
        results = model(str(img_path))
        
        for r in results:
            # Guardar imagen anotada
            annotated_img = r.plot()
            output_path = output_dir / f"predicted_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            
            # Extraer detecciones
            detections = []
            if r.boxes is not None:
                for box in r.boxes:
                    detection = {
                        'class_id': int(box.cls),
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0]
                    }
                    detections.append(detection)
            
            results_summary['results'].append({
                'image': img_path.name,
                'detections': detections
            })
    
    # Guardar resultados
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Resultados guardados en: {output_dir}")
    return results_summary

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python gpu_inference.py <modelo.pt> <batch_dir> <output_dir>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    batch_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    run_inference(model_path, batch_dir, output_dir)
