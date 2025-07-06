#!/usr/bin/env python3
"""
🔍 Model Inspector (Sin PyTorch)
Analiza modelos YOLO sin necesidad de torch/ultralytics
"""

import os
import sys
import zipfile
import json
from pathlib import Path
import yaml
import pickle

class ModelInspector:
    """Inspector de modelos YOLO sin dependencias pesadas."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
    def inspect_model(self):
        """Inspecciona un modelo .pt sin cargar torch."""
        print(f"🔍 Inspeccionando modelo: {self.model_path}")
        
        if not self.model_path.exists():
            print(f"❌ Modelo no encontrado: {self.model_path}")
            return
            
        # Información básica del archivo
        file_size = self.model_path.stat().st_size
        print(f"📁 Tamaño del archivo: {file_size / (1024*1024):.2f} MB")
        
        # Intentar extraer metadata sin torch
        try:
            # Los modelos YOLO a veces tienen metadata legible
            with open(self.model_path, 'rb') as f:
                content = f.read(1024)  # Leer primeros 1KB
                if b'ultralytics' in content:
                    print("✅ Modelo Ultralytics YOLO detectado")
                elif b'yolo' in content.lower():
                    print("✅ Modelo YOLO detectado")
                    
        except Exception as e:
            print(f"⚠️ No se pudo leer metadata: {e}")
            
        return {
            'path': str(self.model_path),
            'size_mb': file_size / (1024*1024),
            'exists': True
        }

class DatasetPreparer:
    """Prepara datos para enviar a máquina con GPU."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
    def create_test_batch(self, output_dir, max_images=10, split="val"):
        """Crea un lote de prueba para enviar a GPU."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creando lote de prueba en: {output_dir}")
        
        # Buscar imágenes
        images_dir = self.dataset_path / split / "images"
        if not images_dir.exists():
            print(f"❌ No se encontró: {images_dir}")
            return
            
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files = image_files[:max_images]
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en: {images_dir}")
            return
            
        # Crear estructura para el lote
        batch_dir = output_dir / f"batch_{split}"
        batch_dir.mkdir(exist_ok=True)
        
        # Copiar imágenes
        import shutil
        copied_files = []
        for i, img_file in enumerate(image_files):
            dest_file = batch_dir / f"test_{i+1:03d}_{img_file.name}"
            shutil.copy2(img_file, dest_file)
            copied_files.append(str(dest_file))
            
        # Crear manifiesto
        manifest = {
            'batch_id': f"batch_{split}_{len(image_files)}",
            'split': split,
            'total_images': len(image_files),
            'image_files': copied_files,
            'dataset_path': str(self.dataset_path),
            'created_at': str(Path(__file__).stat().st_mtime)
        }
        
        # Guardar manifiesto
        with open(batch_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        # Copiar data.yaml si existe
        data_yaml = self.dataset_path / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, batch_dir / "data.yaml")
            
        print(f"✅ Lote creado: {len(image_files)} imágenes")
        print(f"📁 Ubicación: {batch_dir}")
        print(f"📋 Para usar en GPU:")
        print(f"   1. Transfiere la carpeta: {batch_dir}")
        print(f"   2. Ejecuta: python test_model.py --model tu_modelo.pt --batch {batch_dir}")
        
        return batch_dir

class ResultsProcessor:
    """Procesa resultados que vienen de la máquina con GPU."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def process_results(self):
        """Procesa resultados de inferencia."""
        print(f"📊 Procesando resultados en: {self.results_dir}")
        
        if not self.results_dir.exists():
            print(f"❌ Directorio no encontrado: {self.results_dir}")
            return
            
        # Buscar archivos de resultados
        result_files = list(self.results_dir.glob("results*.json")) + list(self.results_dir.glob("*results.json"))
        image_files = list(self.results_dir.glob("predicted_*.jpg")) + list(self.results_dir.glob("predicted_*.png"))
        
        print(f"📄 Archivos de resultados encontrados: {len(result_files)}")
        print(f"🖼️ Imágenes con predicciones: {len(image_files)}")
        
        # Procesar resultados JSON
        all_results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    all_results.append(data)
                    print(f"✅ Procesado: {result_file.name}")
            except Exception as e:
                print(f"❌ Error al leer {result_file.name}: {e}")
                
        # Generar resumen
        if all_results:
            self.generate_summary(all_results)
        else:
            print("⚠️ No se encontraron resultados válidos")
            
        return all_results
    
    def generate_summary(self, results):
        """Genera resumen de los resultados."""
        print(f"\n📊 RESUMEN DE RESULTADOS:")
        
        total_images = 0
        total_detections = 0
        class_counts = {}
        confidences = []
        
        for result in results:
            # Manejar formato directo con 'results'
            if 'results' in result:
                total_images += len(result['results'])
                
                for img_result in result['results']:
                    if 'detections' in img_result:
                        total_detections += len(img_result['detections'])
                        
                        for detection in img_result['detections']:
                            class_name = detection.get('class', 'unknown')
                            confidence = detection.get('confidence', 0)
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = 0
                            class_counts[class_name] += 1
                            confidences.append(confidence)
            
            # Manejar formato legacy con 'detections' directo
            elif 'detections' in result:
                total_detections += len(result['detections'])
                
                for detection in result['detections']:
                    class_name = detection.get('class', 'unknown')
                    confidence = detection.get('confidence', 0)
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    confidences.append(confidence)
        
        print(f"   📸 Total imágenes procesadas: {total_images}")
        print(f"   🎯 Total detecciones: {total_detections}")
        if total_images > 0:
            print(f"   📈 Promedio detecciones/imagen: {total_detections/total_images:.2f}")
        print(f"   🏷️ Clases detectadas:")
        for class_name, count in class_counts.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
            
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            print(f"   📊 Confianza promedio: {avg_conf:.3f}")
            print(f"   📊 Rango confianza: {min_conf:.3f} - {max_conf:.3f}")
        else:
            print("   ⚠️ No se encontraron detecciones")

class GPUScriptGenerator:
    """Genera scripts para ejecutar en la máquina con GPU."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_inference_script(self):
        """Crea script de inferencia para GPU."""
        script_content = '''#!/usr/bin/env python3
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
'''
        
        script_file = self.output_dir / "gpu_inference.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        # Hacer ejecutable
        os.chmod(script_file, 0o755)
        
        print(f"📝 Script creado: {script_file}")
        return script_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Herramientas sin PyTorch para modelos YOLO")
    parser.add_argument("--action", choices=["inspect", "prepare", "process", "generate"], 
                       required=True, help="Acción a realizar")
    parser.add_argument("--model", help="Ruta al modelo (.pt)")
    parser.add_argument("--dataset", help="Ruta al dataset")
    parser.add_argument("--output", help="Directorio de salida")
    parser.add_argument("--results", help="Directorio con resultados de GPU")
    parser.add_argument("--split", default="val", help="Split del dataset")
    parser.add_argument("--max-images", type=int, default=10, help="Máximo de imágenes")
    
    args = parser.parse_args()
    
    if args.action == "inspect":
        if not args.model:
            print("❌ Se requiere --model para inspeccionar")
            return
        inspector = ModelInspector(args.model)
        inspector.inspect_model()
        
    elif args.action == "prepare":
        if not args.dataset or not args.output:
            print("❌ Se requiere --dataset y --output para preparar")
            return
        preparer = DatasetPreparer(args.dataset)
        preparer.create_test_batch(args.output, args.max_images, args.split)
        
    elif args.action == "process":
        if not args.results:
            print("❌ Se requiere --results para procesar")
            return
        processor = ResultsProcessor(args.results)
        processor.process_results()
        
    elif args.action == "generate":
        if not args.output:
            print("❌ Se requiere --output para generar scripts")
            return
        generator = GPUScriptGenerator(args.output)
        generator.create_inference_script()

if __name__ == "__main__":
    main()
