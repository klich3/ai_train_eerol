#!/usr/bin/env python3
"""
🧪 Test YOLO Model
Sistema de prueba para modelos YOLO entrenados
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

class DentalYOLOTester:
    """Tester para modelos YOLO dentales."""
    
    def __init__(self, model_path, dataset_path=None):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model = None
        self.classes = None
        
    def load_model(self):
        """Cargar el modelo YOLO."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        print(f"🔄 Cargando modelo: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"✅ Modelo cargado exitosamente")
        
        # Obtener clases del modelo
        self.classes = self.model.names
        print(f"📋 Clases detectadas: {list(self.classes.values())}")
        
    def load_dataset_info(self):
        """Cargar información del dataset."""
        if not self.dataset_path:
            return
            
        data_yaml = self.dataset_path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            print(f"📊 Dataset info:")
            print(f"   Clases: {data.get('nc', 'N/A')}")
            print(f"   Nombres: {data.get('names', 'N/A')}")
            
    def test_single_image(self, image_path, output_dir=None):
        """Probar el modelo en una imagen individual."""
        if not self.model:
            self.load_model()
            
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        print(f"🖼️ Probando imagen: {image_path.name}")
        
        # Realizar predicción
        results = self.model(str(image_path))
        
        # Procesar resultados
        for r in results:
            # Imagen con anotaciones
            annotated_img = r.plot()
            
            # Guardar resultado si se especifica directorio
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"predicted_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"💾 Resultado guardado: {output_path}")
            
            # Mostrar estadísticas
            boxes = r.boxes
            if boxes is not None:
                print(f"🎯 Detecciones encontradas: {len(boxes)}")
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                    print(f"   {i+1}. {cls_name}: {conf:.3f}")
            else:
                print("❌ No se encontraron detecciones")
                
        return results
    
    def test_dataset(self, split="val", max_images=10, output_dir=None):
        """Probar el modelo en un conjunto de datos."""
        if not self.dataset_path:
            raise ValueError("No se especificó ruta del dataset")
            
        if not self.model:
            self.load_model()
            
        # Buscar imágenes en el split especificado
        images_dir = self.dataset_path / split / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {images_dir}")
            
        # Obtener lista de imágenes
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files = image_files[:max_images]
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en: {images_dir}")
            return
            
        print(f"🧪 Probando {len(image_files)} imágenes del conjunto '{split}'")
        
        results_summary = {
            'total_images': len(image_files),
            'total_detections': 0,
            'class_counts': {},
            'avg_confidence': 0.0
        }
        
        all_confidences = []
        
        for i, img_path in enumerate(image_files):
            print(f"\n📸 Imagen {i+1}/{len(image_files)}: {img_path.name}")
            
            results = self.model(str(img_path))
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    results_summary['total_detections'] += len(boxes)
                    
                    for box in boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                        
                        # Contar clases
                        if cls_name not in results_summary['class_counts']:
                            results_summary['class_counts'][cls_name] = 0
                        results_summary['class_counts'][cls_name] += 1
                        
                        all_confidences.append(conf)
                        
                    # Guardar imagen anotada
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        annotated_img = r.plot()
                        output_path = output_dir / f"test_{i+1:03d}_{img_path.name}"
                        cv2.imwrite(str(output_path), annotated_img)
        
        # Calcular estadísticas
        if all_confidences:
            results_summary['avg_confidence'] = np.mean(all_confidences)
            
        # Mostrar resumen
        print(f"\n📊 RESUMEN DE PRUEBAS:")
        print(f"   Total imágenes: {results_summary['total_images']}")
        print(f"   Total detecciones: {results_summary['total_detections']}")
        print(f"   Promedio de confianza: {results_summary['avg_confidence']:.3f}")
        print(f"   Detecciones por clase:")
        for cls_name, count in results_summary['class_counts'].items():
            print(f"     {cls_name}: {count}")
            
        return results_summary
    
    def benchmark_model(self, test_data_path=None):
        """Ejecutar benchmark completo del modelo."""
        if not self.model:
            self.load_model()
            
        print("🏁 Ejecutando benchmark del modelo...")
        
        # Usar dataset de prueba si está disponible
        if test_data_path or (self.dataset_path and (self.dataset_path / "test").exists()):
            test_path = test_data_path or self.dataset_path
            
            print("🧪 Evaluando en conjunto de prueba...")
            try:
                # Evaluar con métricas YOLO
                results = self.model.val(data=str(self.dataset_path / "data.yaml"), split="test")
                
                print(f"📊 Métricas de evaluación:")
                if hasattr(results, 'box'):
                    print(f"   mAP50: {results.box.map50:.3f}")
                    print(f"   mAP50-95: {results.box.map:.3f}")
                    print(f"   Precisión: {results.box.mp:.3f}")
                    print(f"   Recall: {results.box.mr:.3f}")
                    
            except Exception as e:
                print(f"⚠️ Error en evaluación automática: {e}")
                print("💡 Ejecutando evaluación manual...")
                
                # Evaluación manual
                summary = self.test_dataset(split="test", max_images=20)
                return summary
        else:
            print("⚠️ No se encontró conjunto de prueba, usando validación")
            summary = self.test_dataset(split="val", max_images=20)
            return summary

def main():
    parser = argparse.ArgumentParser(description="Tester para modelos YOLO dentales")
    parser.add_argument("--model", "-m", required=True, help="Ruta al modelo (.pt)")
    parser.add_argument("--dataset", "-d", help="Ruta al dataset")
    parser.add_argument("--image", "-i", help="Probar una imagen específica")
    parser.add_argument("--output", "-o", help="Directorio de salida para resultados")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Ejecutar benchmark completo")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split a usar")
    parser.add_argument("--max-images", type=int, default=10, help="Máximo de imágenes a probar")
    
    args = parser.parse_args()
    
    # Inicializar tester
    tester = DentalYOLOTester(args.model, args.dataset)
    
    try:
        # Cargar modelo
        tester.load_model()
        
        # Cargar info del dataset si está disponible
        if args.dataset:
            tester.load_dataset_info()
        
        # Ejecutar según los argumentos
        if args.image:
            # Probar imagen individual
            results = tester.test_single_image(args.image, args.output)
            
        elif args.benchmark:
            # Ejecutar benchmark completo
            results = tester.benchmark_model()
            
        else:
            # Probar dataset
            if not args.dataset:
                print("❌ Error: Se requiere --dataset para probar conjunto de datos")
                return
                
            results = tester.test_dataset(
                split=args.split, 
                max_images=args.max_images, 
                output_dir=args.output
            )
            
        print("\n🎉 Pruebas completadas exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
