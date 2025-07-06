#!/usr/bin/env python3
"""
🦷 Test Dental Models - Herramienta Automatizada
Prueba modelos dentales de forma simple y automática
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime

# Usar PIL en lugar de OpenCV (más ligero)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("💡 PIL no disponible - las imágenes anotadas serán básicas")

class DentalModelTester:
    """Tester automático para modelos dentales."""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        self.results_dir = self.base_dir / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Clases dentales típicas
        self.dental_classes = [
            "tooth", "caries", "filling", "crown", 
            "implant", "bridge", "root_canal", "wisdom_tooth"
        ]
    
    def find_models(self):
        """Encuentra todos los modelos disponibles."""
        model_files = []
        for pattern in ["*.pt", "*.pth", "*.onnx"]:
            model_files.extend(list(self.models_dir.glob(f"**/{pattern}")))
        return model_files
    
    def find_datasets(self):
        """Encuentra todos los datasets disponibles."""
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                # Verificar si tiene estructura YOLO
                data_yaml = item / "data.yaml"
                val_images = item / "val" / "images"
                test_images = item / "test" / "images"
                
                if data_yaml.exists() or val_images.exists() or test_images.exists():
                    datasets.append(item)
        return datasets
    
    def get_dataset_info(self, dataset_path):
        """Obtiene información del dataset."""
        info = {
            'name': dataset_path.name,
            'path': dataset_path,
            'splits': {},
            'classes': 'N/A'
        }
        
        # Contar imágenes por split
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / "images"
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
                info['splits'][split] = image_count
        
        # Leer info de clases si existe data.yaml
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                info['classes'] = data.get('nc', 'N/A')
            except:
                pass
        
        return info
    
    def select_model(self):
        """Permite seleccionar un modelo."""
        models = self.find_models()
        
        if not models:
            print("❌ No se encontraron modelos")
            print(f"💡 Coloca modelos (.pt, .pth, .onnx) en: {self.models_dir}")
            return None
        
        print("🔍 MODELOS DISPONIBLES:")
        print("=" * 50)
        for i, model in enumerate(models, 1):
            size_mb = model.stat().st_size / (1024*1024)
            rel_path = model.relative_to(self.base_dir)
            print(f"   {i}. {model.name}")
            print(f"      📁 {rel_path}")
            print(f"      📊 {size_mb:.1f} MB")
            print()
        
        while True:
            try:
                choice = input("🎯 Selecciona modelo (número): ").strip()
                if choice == "0" or choice.lower() == "q":
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"✅ Modelo seleccionado: {selected.name}")
                    return selected
                else:
                    print("❌ Número inválido")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                return None
    
    def select_dataset(self):
        """Permite seleccionar un dataset."""
        datasets = self.find_datasets()
        
        if not datasets:
            print("❌ No se encontraron datasets")
            print(f"💡 Coloca datasets en: {self.datasets_dir}")
            return None
        
        print("📊 DATASETS DISPONIBLES:")
        print("=" * 50)
        for i, dataset in enumerate(datasets, 1):
            info = self.get_dataset_info(dataset)
            print(f"   {i}. {info['name']}")
            print(f"      📋 Clases: {info['classes']}")
            for split, count in info['splits'].items():
                print(f"      📸 {split}: {count} imágenes")
            print()
        
        while True:
            try:
                choice = input("🎯 Selecciona dataset (número): ").strip()
                if choice == "0" or choice.lower() == "q":
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    selected = datasets[idx]
                    print(f"✅ Dataset seleccionado: {selected.name}")
                    return selected
                else:
                    print("❌ Número inválido")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                return None
    
    def create_test_batch(self, dataset_path, max_images=10, split="val"):
        """Crea un lote de imágenes para probar."""
        images_dir = dataset_path / split / "images"
        
        if not images_dir.exists():
            # Probar otros splits si val no existe
            for alt_split in ["test", "train"]:
                alt_dir = dataset_path / alt_split / "images"
                if alt_dir.exists():
                    images_dir = alt_dir
                    split = alt_split
                    print(f"💡 Usando split '{split}' en su lugar")
                    break
            
            if not images_dir.exists():
                print(f"❌ No se encontraron imágenes en: {dataset_path}")
                return []
        
        # Obtener lista de imágenes
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en: {images_dir}")
            return []
        
        # Seleccionar muestra aleatoria
        random.shuffle(image_files)
        selected_images = image_files[:max_images]
        
        print(f"📦 Lote creado: {len(selected_images)} imágenes del split '{split}'")
        return selected_images
    
    def simulate_model_inference(self, model_path, image_files):
        """Simula la inferencia del modelo (para demo sin CUDA)."""
        print(f"🤖 Simulando inferencia con: {model_path.name}")
        print("💡 (En máquina con GPU usarías YOLO real)")
        
        results = {
            'model_info': {
                'name': model_path.name,
                'path': str(model_path),
                'size_mb': model_path.stat().st_size / (1024*1024)
            },
            'inference_info': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(image_files),
                'mode': 'simulation'
            },
            'results': []
        }
        
        for i, img_file in enumerate(image_files):
            # Simular detecciones aleatorias
            num_detections = random.randint(0, 5)
            detections = []
            
            for j in range(num_detections):
                detection = {
                    'class_id': random.randint(0, len(self.dental_classes)-1),
                    'class': random.choice(self.dental_classes),
                    'confidence': round(random.uniform(0.3, 0.95), 3),
                    'bbox': [
                        random.randint(50, 400),   # x1
                        random.randint(50, 300),   # y1
                        random.randint(450, 800),  # x2
                        random.randint(350, 600)   # y2
                    ]
                }
                detections.append(detection)
            
            results['results'].append({
                'image': img_file.name,
                'image_path': str(img_file),
                'detections': detections
            })
            
            print(f"  📸 {i+1}/{len(image_files)}: {img_file.name} → {num_detections} detecciones")
        
        return results
    
    def create_annotated_images(self, results, output_dir):
        """Crea imágenes con las detecciones marcadas."""
        output_dir = Path(output_dir)
        images_dir = output_dir / "annotated_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Creando imágenes anotadas en: {images_dir}")
        
        if not HAS_PIL:
            print("⚠️ PIL no disponible - creando archivos de texto con detecciones")
            return self.create_text_annotations(results, images_dir)
        
        colors = [
            (0, 255, 0),    # Verde
            (255, 0, 0),    # Rojo  
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarillo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Púrpura
            (255, 165, 0)   # Naranja
        ]
        
        annotated_count = 0
        
        for result in results['results']:
            img_path = Path(result['image_path'])
            detections = result['detections']
            
            if not img_path.exists():
                print(f"⚠️ Imagen no encontrada: {img_path}")
                continue
            
            try:
                # Cargar imagen con PIL
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                
                # Usar fuente por defecto
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Dibujar detecciones
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    x1, y1, x2, y2 = bbox
                    color = colors[i % len(colors)]
                    
                    # Dibujar rectángulo
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Dibujar etiqueta
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Dibujar fondo para texto
                    bbox_text = draw.textbbox((x1, y1-20), label, font=font) if font else (x1, y1-20, x1+100, y1)
                    draw.rectangle(bbox_text, fill=color)
                    
                    # Dibujar texto
                    draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)
                
                # Guardar imagen anotada
                output_path = images_dir / f"annotated_{img_path.name}"
                img.save(output_path)
                annotated_count += 1
                
            except Exception as e:
                print(f"❌ Error procesando {img_path.name}: {e}")
        
        print(f"✅ Creadas {annotated_count} imágenes anotadas")
        return annotated_count
    
    def create_text_annotations(self, results, output_dir):
        """Crea archivos de texto con las detecciones cuando no hay PIL."""
        print("📝 Creando anotaciones en formato texto...")
        
        annotation_count = 0
        
        for result in results['results']:
            img_name = result['image']
            detections = result['detections']
            
            # Crear archivo de texto con detecciones
            txt_file = output_dir / f"detections_{Path(img_name).stem}.txt"
            
            with open(txt_file, 'w') as f:
                f.write(f"DETECCIONES PARA: {img_name}\n")
                f.write("=" * 50 + "\n\n")
                
                if not detections:
                    f.write("❌ No se encontraron detecciones\n")
                else:
                    for i, detection in enumerate(detections, 1):
                        f.write(f"Detección {i}:\n")
                        f.write(f"  Clase: {detection['class']}\n")
                        f.write(f"  Confianza: {detection['confidence']:.3f}\n")
                        f.write(f"  Coordenadas: {detection['bbox']}\n")
                        f.write("\n")
                        
                    f.write(f"\nTOTAL: {len(detections)} detecciones encontradas\n")
            
            annotation_count += 1
            
            # También copiar imagen original
            try:
                img_path = Path(result['image_path'])
                if img_path.exists():
                    dest_path = output_dir / f"original_{img_name}"
                    shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"⚠️ No se pudo copiar imagen: {e}")
        
        print(f"✅ Creados {annotation_count} archivos de anotaciones")
        return annotation_count
    
    def generate_test_report(self, results, output_dir):
        """Genera reporte de la prueba."""
        output_dir = Path(output_dir)
        
        # Calcular estadísticas
        total_images = len(results['results'])
        total_detections = sum(len(r['detections']) for r in results['results'])
        
        class_counts = {}
        confidences = []
        
        for result in results['results']:
            for detection in result['detections']:
                class_name = detection['class']
                confidence = detection['confidence']
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidences.append(confidence)
        
        # Crear reporte
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'model': results['model_info'],
                'total_images': total_images,
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / max(total_images, 1),
                'class_distribution': class_counts,
                'confidence_stats': {
                    'avg': sum(confidences) / max(len(confidences), 1) if confidences else 0,
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0,
                    'count': len(confidences)
                }
            },
            'detailed_results': results
        }
        
        # Guardar JSON
        report_file = output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN DE LA PRUEBA:")
        print("=" * 50)
        print(f"🤖 Modelo: {results['model_info']['name']}")
        print(f"📸 Imágenes procesadas: {total_images}")
        print(f"🎯 Total detecciones: {total_detections}")
        print(f"📈 Promedio por imagen: {total_detections / max(total_images, 1):.2f}")
        
        if confidences:
            print(f"📊 Confianza promedio: {sum(confidences) / len(confidences):.3f}")
            print(f"📊 Rango: {min(confidences):.3f} - {max(confidences):.3f}")
        
        print(f"\n🏷️ CLASES DETECTADAS:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n💾 Reporte completo: {report_file}")
        return report_file
    
    def run_complete_test(self):
        """Ejecuta una prueba completa automática."""
        print("🦷 DENTAL MODEL TESTER - Prueba Automática")
        print("=" * 60)
        
        # 1. Seleccionar modelo
        print("\n1️⃣ SELECCIÓN DE MODELO:")
        model_path = self.select_model()
        if not model_path:
            print("❌ Prueba cancelada")
            return
        
        # 2. Seleccionar dataset
        print(f"\n2️⃣ SELECCIÓN DE DATASET:")
        dataset_path = self.select_dataset()
        if not dataset_path:
            print("❌ Prueba cancelada")
            return
        
        # 3. Configurar prueba
        print(f"\n3️⃣ CONFIGURACIÓN DE PRUEBA:")
        try:
            max_images = int(input("📸 Número máximo de imágenes a probar (default 10): ") or "10")
        except ValueError:
            max_images = 10
        
        # 4. Crear lote de prueba
        print(f"\n4️⃣ CREANDO LOTE DE PRUEBA:")
        image_files = self.create_test_batch(dataset_path, max_images)
        if not image_files:
            print("❌ No se pudo crear el lote")
            return
        
        # 5. Crear directorio de resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"{model_path.stem}_{dataset_path.name}_{timestamp}"
        output_dir = self.results_dir / test_name
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n5️⃣ EJECUTANDO INFERENCIA:")
        print(f"📁 Resultados en: {output_dir}")
        
        # 6. Simular inferencia
        results = self.simulate_model_inference(model_path, image_files)
        
        # 7. Crear imágenes anotadas
        print(f"\n6️⃣ CREANDO IMÁGENES ANOTADAS:")
        self.create_annotated_images(results, output_dir)
        
        # 8. Generar reporte
        print(f"\n7️⃣ GENERANDO REPORTE:")
        report_file = self.generate_test_report(results, output_dir)
        
        # 9. Mostrar ubicaciones
        print(f"\n🎉 PRUEBA COMPLETADA:")
        print(f"📁 Resultados: {output_dir}")
        print(f"🖼️ Imágenes anotadas: {output_dir / 'annotated_images'}")
        print(f"📄 Reporte JSON: {report_file}")
        
        # 10. Preguntar si abrir resultados
        if input("\n🔍 ¿Abrir carpeta de resultados? (s/N): ").lower().startswith('s'):
            try:
                if sys.platform == "darwin":  # macOS
                    os.system(f"open '{output_dir}'")
                elif sys.platform == "win32":  # Windows
                    os.system(f"explorer '{output_dir}'")
                else:  # Linux
                    os.system(f"xdg-open '{output_dir}'")
            except:
                print(f"💡 Abre manualmente: {output_dir}")

def main():
    """Función principal."""
    try:
        tester = DentalModelTester()
        tester.run_complete_test()
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
