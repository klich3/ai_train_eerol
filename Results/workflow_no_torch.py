#!/usr/bin/env python3
"""
🔄 Workflow Manager (Sin PyTorch)
Gestiona el flujo de trabajo entre máquina local y GPU
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

class NoTorchWorkflow:
    """Gestor de workflow sin PyTorch."""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        self.batches_dir = self.base_dir / "batches"
        self.results_dir = self.base_dir / "results"
        
        # Crear directorios necesarios
        for dir_path in [self.batches_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self):
        """Lista modelos disponibles."""
        print("🔍 Buscando modelos disponibles...")
        
        model_files = []
        for ext in ["*.pt", "*.pth", "*.onnx"]:
            model_files.extend(list(self.models_dir.glob(f"**/{ext}")))
            
        if not model_files:
            print("❌ No se encontraron modelos")
            print(f"💡 Busca en: {self.models_dir}")
            return []
            
        print(f"✅ Encontrados {len(model_files)} modelos:")
        for i, model in enumerate(model_files, 1):
            size_mb = model.stat().st_size / (1024*1024)
            print(f"   {i}. {model.name} ({size_mb:.1f} MB)")
            print(f"      📁 {model.parent}")
            
        return model_files
    
    def list_available_datasets(self):
        """Lista datasets disponibles."""
        print("📊 Buscando datasets disponibles...")
        
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                data_yaml = item / "data.yaml"
                if data_yaml.exists():
                    datasets.append(item)
                    
        if not datasets:
            print("❌ No se encontraron datasets")
            print(f"💡 Busca en: {self.datasets_dir}")
            return []
            
        print(f"✅ Encontrados {len(datasets)} datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i}. {dataset.name}")
            
            # Leer info del data.yaml
            try:
                import yaml
                with open(dataset / "data.yaml", 'r') as f:
                    data = yaml.safe_load(f)
                print(f"      📋 Clases: {data.get('nc', 'N/A')}")
                
                # Contar imágenes
                for split in ['train', 'val', 'test']:
                    split_dir = dataset / split / "images"
                    if split_dir.exists():
                        count = len(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
                        print(f"      📸 {split}: {count} imágenes")
                        
            except Exception as e:
                print(f"      ⚠️ Error leyendo metadata: {e}")
                
        return datasets
    
    def create_test_batch(self, model_name, dataset_name, max_images=10, splits=["val"]):
        """Crea un lote de prueba."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"{model_name}_{dataset_name}_{timestamp}"
        batch_dir = self.batches_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creando lote: {batch_id}")
        
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset no encontrado: {dataset_path}")
            return None
            
        batch_info = {
            'batch_id': batch_id,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'created_at': timestamp,
            'max_images': max_images,
            'splits': splits,
            'total_images': 0,
            'files': []
        }
        
        total_copied = 0
        
        for split in splits:
            split_dir = dataset_path / split / "images"
            if not split_dir.exists():
                print(f"⚠️ Split no encontrado: {split}")
                continue
                
            # Crear directorio para este split
            batch_split_dir = batch_dir / split
            batch_split_dir.mkdir(exist_ok=True)
            
            # Buscar imágenes
            image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            image_files = image_files[:max_images]
            
            print(f"📸 Copiando {len(image_files)} imágenes de {split}...")
            
            for i, img_file in enumerate(image_files):
                dest_file = batch_split_dir / f"{split}_{i+1:03d}_{img_file.name}"
                shutil.copy2(img_file, dest_file)
                batch_info['files'].append(str(dest_file.relative_to(batch_dir)))
                total_copied += 1
        
        batch_info['total_images'] = total_copied
        
        # Copiar data.yaml
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, batch_dir / "data.yaml")
            
        # Guardar información del lote
        with open(batch_dir / "batch_info.json", 'w') as f:
            json.dump(batch_info, f, indent=2)
            
        # Crear README con instrucciones
        readme_content = f"""# Lote de Prueba: {batch_id}

## 📋 Información
- **Modelo**: {model_name}
- **Dataset**: {dataset_name}
- **Creado**: {timestamp}
- **Total imágenes**: {total_copied}

## 🚀 Instrucciones para GPU

1. **Transferir este directorio** a la máquina con GPU
2. **Instalar dependencias** (si no están):
   ```bash
   pip install ultralytics opencv-python
   ```
3. **Ejecutar inferencia**:
   ```bash
   python gpu_inference.py path/to/model.pt {batch_id} output_results/
   ```
4. **Transferir resultados** de vuelta a la máquina local

## 📁 Estructura
```
{batch_id}/
├── batch_info.json    # Información del lote
├── data.yaml         # Configuración del dataset
├── README.md         # Este archivo
└── val/              # Imágenes de validación
    ├── val_001_*.jpg
    ├── val_002_*.jpg
    └── ...
```

## 🔄 Procesamiento Local
Una vez que tengas los resultados:
```bash
python model_tools_no_torch.py --action process --results path/to/results/
```
"""
        
        with open(batch_dir / "README.md", 'w') as f:
            f.write(readme_content)
            
        print(f"✅ Lote creado: {batch_dir}")
        print(f"📋 Total imágenes: {total_copied}")
        print(f"📁 Para transferir: {batch_dir}")
        
        return batch_dir
    
    def list_batches(self):
        """Lista lotes creados."""
        print("📦 Lotes disponibles:")
        
        batches = list(self.batches_dir.glob("*/batch_info.json"))
        
        if not batches:
            print("❌ No se encontraron lotes")
            return []
            
        batch_info_list = []
        for i, batch_file in enumerate(batches, 1):
            try:
                with open(batch_file, 'r') as f:
                    info = json.load(f)
                    
                batch_info_list.append(info)
                print(f"   {i}. {info['batch_id']}")
                print(f"      📋 Modelo: {info['model_name']}")
                print(f"      📊 Dataset: {info['dataset_name']}")
                print(f"      📸 Imágenes: {info['total_images']}")
                print(f"      🕒 Creado: {info['created_at']}")
                
            except Exception as e:
                print(f"   ❌ Error leyendo {batch_file}: {e}")
                
        return batch_info_list
    
    def process_results(self, results_dir):
        """Procesa resultados de GPU."""
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            print(f"❌ Directorio de resultados no encontrado: {results_dir}")
            return
            
        print(f"📊 Procesando resultados en: {results_dir}")
        
        # Buscar archivos de resultados
        result_files = list(results_dir.glob("results*.json"))
        image_files = list(results_dir.glob("predicted_*.jpg")) + list(results_dir.glob("predicted_*.png"))
        
        print(f"📄 Archivos JSON: {len(result_files)}")
        print(f"🖼️ Imágenes predichas: {len(image_files)}")
        
        if not result_files:
            print("❌ No se encontraron archivos de resultados JSON")
            return
            
        # Procesar cada archivo de resultados
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
        self.generate_summary_report(all_results, results_dir)
        
        return all_results
    
    def generate_summary_report(self, results, output_dir):
        """Genera reporte resumen."""
        print(f"\n📊 GENERANDO REPORTE RESUMEN...")
        
        total_images = 0
        total_detections = 0
        class_counts = {}
        confidences = []
        
        for result_set in results:
            if 'results' in result_set:
                total_images += len(result_set['results'])
                
                for result in result_set['results']:
                    if 'detections' in result:
                        total_detections += len(result['detections'])
                        
                        for detection in result['detections']:
                            class_name = detection.get('class', 'unknown')
                            confidence = detection.get('confidence', 0)
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = 0
                            class_counts[class_name] += 1
                            confidences.append(confidence)
        
        # Crear reporte
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_images': total_images,
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / max(total_images, 1),
                'class_distribution': class_counts,
                'avg_confidence': sum(confidences) / max(len(confidences), 1) if confidences else 0,
                'confidence_stats': {
                    'min': min(confidences) if confididences else 0,
                    'max': max(confidences) if confidences else 0,
                    'count': len(confidences)
                }
            },
            'raw_results': results
        }
        
        # Guardar reporte
        report_file = Path(output_dir) / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Mostrar resumen en consola
        print(f"\n📊 RESUMEN DE RESULTADOS:")
        print(f"   📸 Total imágenes: {total_images}")
        print(f"   🎯 Total detecciones: {total_detections}")
        print(f"   📈 Promedio por imagen: {total_detections / max(total_images, 1):.2f}")
        print(f"   🏷️ Clases detectadas:")
        for class_name, count in class_counts.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        if confidences:
            print(f"   📊 Confianza promedio: {sum(confidences) / len(confidences):.3f}")
            
        print(f"\n💾 Reporte guardado: {report_file}")
    
    def list_processed_results(self):
        """Lista resultados procesados disponibles."""
        print("🔍 Buscando resultados procesados...")
        
        # Buscar en directorio de resultados
        result_dirs = []
        for item in self.results_dir.iterdir():
            if item.is_dir():
                # Buscar archivos de resumen
                summary_files = list(item.glob("summary_report.json"))
                result_files = list(item.glob("results*.json"))
                
                if summary_files or result_files:
                    result_dirs.append(item)
        
        if not result_dirs:
            print("❌ No se encontraron resultados procesados")
            print(f"💡 Busca en: {self.results_dir}")
            return []
            
        print(f"✅ Encontrados {len(result_dirs)} directorios con resultados:")
        for i, result_dir in enumerate(result_dirs, 1):
            print(f"   {i}. {result_dir.name}")
            
            # Mostrar info básica si existe resumen
            summary_file = result_dir / "summary_report.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    stats = summary.get('summary', {})
                    print(f"      📸 Imágenes: {stats.get('total_images', 'N/A')}")
                    print(f"      🎯 Detecciones: {stats.get('total_detections', 'N/A')}")
                    print(f"      📊 Confianza: {stats.get('avg_confidence', 0):.3f}")
                except:
                    pass
                    
        return result_dirs
    
    def show_detailed_results(self, results_dir):
        """Muestra resultados detallados de un directorio."""
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            print(f"❌ Directorio no encontrado: {results_dir}")
            return
            
        print(f"📊 RESULTADOS DETALLADOS: {results_dir.name}")
        print("=" * 50)
        
        # Leer resumen si existe
        summary_file = results_dir / "summary_report.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                stats = summary.get('summary', {})
                print(f"📊 ESTADÍSTICAS GENERALES:")
                print(f"   📸 Total imágenes: {stats.get('total_images', 'N/A')}")
                print(f"   🎯 Total detecciones: {stats.get('total_detections', 'N/A')}")
                print(f"   📈 Promedio por imagen: {stats.get('avg_detections_per_image', 0):.2f}")
                print(f"   📊 Confianza promedio: {stats.get('avg_confidence', 0):.3f}")
                
                conf_stats = stats.get('confidence_stats', {})
                if conf_stats:
                    print(f"   📊 Rango confianza: {conf_stats.get('min', 0):.3f} - {conf_stats.get('max', 0):.3f}")
                
                # Distribución de clases
                class_dist = stats.get('class_distribution', {})
                if class_dist:
                    print(f"\n🏷️ DISTRIBUCIÓN DE CLASES:")
                    total_detections = stats.get('total_detections', 0)
                    for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_detections * 100) if total_detections > 0 else 0
                        print(f"   {class_name}: {count} detecciones ({percentage:.1f}%)")
                        
            except Exception as e:
                print(f"❌ Error leyendo resumen: {e}")
        
        # Listar archivos de imágenes disponibles
        image_files = list(results_dir.glob("predicted_*.jpg")) + list(results_dir.glob("predicted_*.png"))
        if image_files:
            print(f"\n🖼️ IMÁGENES PROCESADAS ({len(image_files)}):")
            for i, img_file in enumerate(image_files[:10], 1):  # Mostrar primeras 10
                print(f"   {i}. {img_file.name}")
            if len(image_files) > 10:
                print(f"   ... y {len(image_files) - 10} más")
    
    def show_images_with_detections(self, results_dir):
        """Intenta mostrar imágenes con detecciones usando herramientas del sistema."""
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            print(f"❌ Directorio no encontrado: {results_dir}")
            return
            
        # Buscar imágenes procesadas
        image_files = list(results_dir.glob("predicted_*.jpg")) + list(results_dir.glob("predicted_*.png"))
        
        if not image_files:
            print(f"❌ No se encontraron imágenes procesadas en: {results_dir}")
            return
            
        print(f"🖼️ Encontradas {len(image_files)} imágenes procesadas")
        print("🔍 Opciones de visualización:")
        print("   1. 📁 Abrir carpeta en Finder/Explorer")
        print("   2. 🖼️ Abrir primera imagen")
        print("   3. 📋 Listar todas las imágenes")
        print("   0. ⬅️ Volver")
        
        choice = input("\n🎯 Selecciona opción: ").strip()
        
        if choice == "1":
            # Abrir carpeta
            import subprocess
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", str(results_dir)], check=True)
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", str(results_dir)], check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", str(results_dir)], check=True)
                print(f"✅ Carpeta abierta: {results_dir}")
            except Exception as e:
                print(f"❌ Error abriendo carpeta: {e}")
                
        elif choice == "2":
            # Abrir primera imagen
            import subprocess
            try:
                first_image = image_files[0]
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", str(first_image)], check=True)
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", str(first_image)], shell=True, check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", str(first_image)], check=True)
                print(f"✅ Imagen abierta: {first_image.name}")
            except Exception as e:
                print(f"❌ Error abriendo imagen: {e}")
                
        elif choice == "3":
            # Listar todas las imágenes
            print(f"\n📋 TODAS LAS IMÁGENES PROCESADAS:")
            for i, img_file in enumerate(image_files, 1):
                size_kb = img_file.stat().st_size / 1024
                print(f"   {i:2d}. {img_file.name} ({size_kb:.1f} KB)")
    
    def generate_statistics_report(self, results_dir):
        """Genera estadísticas avanzadas de los resultados."""
        results_dir = Path(results_dir)
        
        # Leer resumen existente
        summary_file = results_dir / "summary_report.json"
        if not summary_file.exists():
            print(f"❌ No se encontró resumen en: {results_dir}")
            print("💡 Ejecuta primero 'Procesar resultados de GPU'")
            return
            
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except Exception as e:
            print(f"❌ Error leyendo resumen: {e}")
            return
            
        stats = summary.get('summary', {})
        raw_results = summary.get('raw_results', [])
        
        print(f"📈 ESTADÍSTICAS AVANZADAS: {results_dir.name}")
        print("=" * 60)
        
        # Estadísticas básicas
        total_images = stats.get('total_images', 0)
        total_detections = stats.get('total_detections', 0)
        
        print(f"📊 RESUMEN GENERAL:")
        print(f"   📸 Total imágenes procesadas: {total_images}")
        print(f"   🎯 Total detecciones: {total_detections}")
        print(f"   📈 Detecciones por imagen: {total_detections / max(total_images, 1):.2f}")
        
        # Análisis de confianza
        conf_stats = stats.get('confidence_stats', {})
        if conf_stats:
            print(f"\n📊 ANÁLISIS DE CONFIANZA:")
            print(f"   🎯 Promedio: {stats.get('avg_confidence', 0):.3f}")
            print(f"   📊 Mínima: {conf_stats.get('min', 0):.3f}")
            print(f"   📊 Máxima: {conf_stats.get('max', 0):.3f}")
            print(f"   📊 Total mediciones: {conf_stats.get('count', 0)}")
        
        # Distribución de clases
        class_dist = stats.get('class_distribution', {})
        if class_dist:
            print(f"\n🏷️ DISTRIBUCIÓN DETALLADA DE CLASES:")
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, count in sorted_classes:
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                bar_length = int(percentage / 5)  # Barra visual
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {class_name:12s}: {count:3d} [{bar}] {percentage:5.1f}%")
        
        # Análisis por imagen (si hay datos detallados)
        if raw_results:
            detections_per_image = []
            for result_set in raw_results:
                if 'results' in result_set:
                    for result in result_set['results']:
                        det_count = len(result.get('detections', []))
                        detections_per_image.append(det_count)
            
            if detections_per_image:
                print(f"\n📈 ANÁLISIS POR IMAGEN:")
                print(f"   📊 Mínimo detecciones: {min(detections_per_image)}")
                print(f"   📊 Máximo detecciones: {max(detections_per_image)}")
                print(f"   📊 Promedio: {sum(detections_per_image) / len(detections_per_image):.2f}")
                
                # Histograma simple
                print(f"\n📊 DISTRIBUCIÓN DE DETECCIONES:")
                max_det = max(detections_per_image) if detections_per_image else 0
                for i in range(max_det + 1):
                    count = detections_per_image.count(i)
                    if count > 0:
                        bar = "█" * (count * 20 // len(detections_per_image))
                        print(f"   {i:2d} det: {count:2d} imgs {bar}")
        
        print(f"\n💾 Datos completos en: {summary_file}")

def main():
    workflow = NoTorchWorkflow()
    
    print("🔄 WORKFLOW MANAGER (Sin PyTorch)")
    print("=" * 50)
    
    while True:
        print("\n🎛️ OPCIONES:")
        print("1. 📋 Listar modelos disponibles")
        print("2. 📊 Listar datasets disponibles") 
        print("3. 📦 Crear lote de prueba")
        print("4. 📦 Listar lotes creados")
        print("5. 📊 Procesar resultados de GPU")
        print("6. 🔍 Ver resultados procesados")
        print("7. 🖼️ Visualizar imágenes con detecciones")
        print("8. 📈 Estadísticas detalladas")
        print("0. 🚪 Salir")
        
        try:
            choice = input("\n🎯 Selecciona una opción: ").strip()
            
            if choice == "0":
                print("👋 ¡Hasta luego!")
                break
                
            elif choice == "1":
                workflow.list_available_models()
                
            elif choice == "2":
                workflow.list_available_datasets()
                
            elif choice == "3":
                # Crear lote interactivo
                models = workflow.list_available_models()
                if not models:
                    continue
                    
                try:
                    model_idx = int(input("Selecciona modelo (número): ")) - 1
                    model_name = models[model_idx].stem
                except (ValueError, IndexError):
                    print("❌ Selección inválida")
                    continue
                
                datasets = workflow.list_available_datasets()
                if not datasets:
                    continue
                    
                try:
                    dataset_idx = int(input("Selecciona dataset (número): ")) - 1
                    dataset_name = datasets[dataset_idx].name
                except (ValueError, IndexError):
                    print("❌ Selección inválida")
                    continue
                
                max_images = input("Máximo de imágenes por split (10): ").strip()
                max_images = int(max_images) if max_images else 10
                
                workflow.create_test_batch(model_name, dataset_name, max_images)
                
            elif choice == "4":
                workflow.list_batches()
                
            elif choice == "5":
                results_dir = input("Directorio con resultados de GPU: ").strip()
                if results_dir:
                    workflow.process_results(results_dir)
                else:
                    print("❌ Directorio requerido")
                    
            elif choice == "6":
                # Ver resultados procesados
                result_dirs = workflow.list_processed_results()
                if result_dirs:
                    try:
                        selection = input("\n🎯 Selecciona resultado (número o nombre): ").strip()
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(result_dirs):
                                workflow.show_detailed_results(result_dirs[idx])
                            else:
                                print("❌ Número inválido")
                        else:
                            # Buscar por nombre
                            found = False
                            for result_dir in result_dirs:
                                if selection.lower() in result_dir.name.lower():
                                    workflow.show_detailed_results(result_dir)
                                    found = True
                                    break
                            if not found:
                                print("❌ Resultado no encontrado")
                    except ValueError:
                        print("❌ Entrada inválida")
                        
            elif choice == "7":
                # Visualizar imágenes
                result_dirs = workflow.list_processed_results()
                if result_dirs:
                    try:
                        selection = input("\n🎯 Selecciona resultado para ver imágenes (número): ").strip()
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(result_dirs):
                                workflow.show_images_with_detections(result_dirs[idx])
                            else:
                                print("❌ Número inválido")
                        else:
                            print("❌ Ingresa un número")
                    except ValueError:
                        print("❌ Entrada inválida")
                        
            elif choice == "8":
                # Estadísticas detalladas
                result_dirs = workflow.list_processed_results()
                if result_dirs:
                    try:
                        selection = input("\n🎯 Selecciona resultado para estadísticas (número): ").strip()
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(result_dirs):
                                workflow.generate_statistics_report(result_dirs[idx])
                            else:
                                print("❌ Número inválido")
                        else:
                            print("❌ Ingresa un número")
                    except ValueError:
                        print("❌ Entrada inválida")
                
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
