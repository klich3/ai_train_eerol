# 🦷 Ejemplos de Uso - Dental AI Workflow Manager

## 🚀 Inicio Rápido

### Análisis Básico de Datasets
```python
# Análisis rápido de todos los datasets
from Utils.advanced_analysis import analyze_dental_datasets

# Ejecutar análisis completo
results = analyze_dental_datasets("_dataSets", "StatisticsResults")

# Los resultados se guardan automáticamente en StatisticsResults/
print(f"Análisis completado: {results['summary']['total_images']} imágenes analizadas")
```

### Workflow Completo
```python
from Src.workflow_manager import DentalWorkflowManager

# Crear manager
workflow = DentalWorkflowManager(
    source_dir="_dataSets",
    output_dir="Dist/dental_ai"
)

# Ejecutar workflow completo
workflow.run_complete_workflow()
```

## 📊 Ejemplos de Análisis

### 1. Análisis de Calidad de Imágenes
```python
from Utils.advanced_analysis import AdvancedDatasetAnalyzer

analyzer = AdvancedDatasetAnalyzer("_dataSets", "StatisticsResults")

# Análisis específico de calidad
quality_results = analyzer.analyze_image_quality()

# Visualizar resultados
analyzer.create_quality_analysis()
```

### 2. Análisis de Distribución de Clases
```python
# Cargar resultados de análisis previo
with open("StatisticsResults/dental_dataset_analysis.json") as f:
    data = json.load(f)

# Extraer distribución de clases
class_distribution = {}
for dataset_type, info in data.items():
    if dataset_type == 'summary':
        continue
    
    for dataset_name, dataset_info in info.get('datasets', {}).items():
        classes = dataset_info.get('class_distribution', {})
        for class_name, count in classes.items():
            class_distribution[class_name] = class_distribution.get(class_name, 0) + count

print("Distribución de clases:")
for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {count} instancias")
```

### 3. Crear Dashboard Personalizado
```python
from Utils.visualization import DatasetVisualizer

visualizer = DatasetVisualizer("StatisticsResults/dental_dataset_analysis.json")

# Crear visualizaciones personalizadas
visualizer.create_custom_dashboard({
    'title': 'Mi Dashboard Dental',
    'sections': ['overview', 'quality', 'distribution'],
    'output_file': 'StatisticsResults/mi_dashboard.html'
})
```

## 🏗️ Ejemplos de Estructura de Proyecto

### Crear Estructura Base
```python
from Src.structure_generator import ProjectStructureGenerator

generator = ProjectStructureGenerator("Dist/dental_ai")

# Crear estructura completa
structure = generator.create_complete_structure({
    'project_name': 'dental_classification',
    'architectures': ['yolo', 'coco', 'unet'],
    'include_api': True,
    'include_docs': True
})

print(f"Estructura creada en: {structure['base_path']}")
```

### Configurar Proyecto Específico
```python
# Configuración para detección de caries
config = {
    'project_name': 'caries_detection',
    'target_classes': ['caries', 'healthy_tooth', 'filling'],
    'image_size': 640,
    'batch_size': 16,
    'epochs': 100
}

generator.create_yolo_project(config)
```

## 🔄 Ejemplos de Procesamiento

### Conversión de Formatos
```python
from Utils.dental_format_converter import DentalFormatConverter

converter = DentalFormatConverter()

# COCO a YOLO
converter.coco_to_yolo(
    coco_dir="datasets/coco_dental",
    output_dir="datasets/yolo_dental",
    class_mapping={'tooth': 0, 'caries': 1}
)

# YOLO a COCO
converter.yolo_to_coco(
    yolo_dir="datasets/yolo_dental",
    output_file="datasets/dental_coco.json",
    image_info={'width': 640, 'height': 640}
)
```

### Augmentación de Datos
```python
from Utils.data_augmentation import DataBalancer

balancer = DataBalancer()

# Balancear dataset desbalanceado
balancer.balance_dataset(
    dataset_dir="datasets/imbalanced_dental",
    output_dir="datasets/balanced_dental",
    target_samples_per_class=1000,
    augmentation_techniques=['rotation', 'flip', 'brightness']
)
```

## 🎯 Ejemplos por Arquitectura

### YOLO Object Detection
```python
# Preparar dataset para YOLO
from Src.data_processor import YOLOProcessor

processor = YOLOProcessor()

# Procesar dataset
yolo_dataset = processor.create_yolo_dataset(
    source_dir="_dataSets/_YOLO",
    output_dir="Dist/dental_ai/yolo",
    classes=['tooth', 'caries', 'implant'],
    train_split=0.8
)

# Generar data.yaml
processor.generate_data_yaml(yolo_dataset, "Dist/dental_ai/yolo/data.yaml")
```

### COCO Segmentation
```python
from Src.data_processor import COCOProcessor

processor = COCOProcessor()

# Crear dataset COCO
coco_dataset = processor.create_coco_dataset(
    source_dir="_dataSets/_COCO",
    output_dir="Dist/dental_ai/coco",
    merge_annotations=True
)

# Validar anotaciones
validation_report = processor.validate_coco_annotations(
    "Dist/dental_ai/coco/annotations/instances_train2017.json"
)
```

### U-Net Segmentation
```python
from Src.data_processor import UNetProcessor

processor = UNetProcessor()

# Preparar para U-Net
unet_dataset = processor.create_unet_dataset(
    images_dir="_dataSets/_pure images and masks",
    masks_dir="_dataSets/_pure images and masks",
    output_dir="Dist/dental_ai/unet",
    image_size=(512, 512)
)
```

## 📈 Ejemplos de Análisis Avanzado

### Análisis de Rendimiento de Datasets
```python
from Utils.advanced_analysis import DatasetPerformanceAnalyzer

analyzer = DatasetPerformanceAnalyzer()

# Evaluar calidad de datasets
performance_report = analyzer.evaluate_dataset_quality(
    dataset_dirs=["_dataSets/_YOLO", "_dataSets/_COCO"],
    metrics=['image_quality', 'annotation_consistency', 'class_balance']
)

# Generar recomendaciones
recommendations = analyzer.generate_improvement_recommendations(performance_report)

print("Recomendaciones de mejora:")
for rec in recommendations:
    print(f"- {rec}")
```

### Comparación de Datasets
```python
# Comparar múltiples datasets
comparison_results = analyzer.compare_datasets([
    {"name": "Dataset A", "path": "_dataSets/_YOLO/dataset_a"},
    {"name": "Dataset B", "path": "_dataSets/_YOLO/dataset_b"},
    {"name": "Dataset C", "path": "_dataSets/_COCO/dataset_c"}
])

# Visualizar comparación
analyzer.create_comparison_chart(comparison_results, "StatisticsResults/comparison.png")
```

## 🔧 Ejemplos de Personalización

### Configuración Personalizada
```python
# Crear configuración personalizada
custom_config = {
    'dataset_config': {
        'supported_formats': ['.jpg', '.png', '.tiff'],
        'target_resolution': (1024, 1024),
        'quality_threshold': 0.7
    },
    'analysis_config': {
        'include_image_analysis': True,
        'include_annotation_analysis': True,
        'generate_visualizations': True,
        'output_formats': ['json', 'csv', 'html']
    },
    'processing_config': {
        'batch_size': 32,
        'num_workers': 4,
        'use_gpu': True
    }
}

# Aplicar configuración
workflow = DentalWorkflowManager(config=custom_config)
```

### Filtros Personalizados
```python
# Filtrar datasets por criterios específicos
def custom_dataset_filter(dataset_info):
    """Filtro personalizado para datasets."""
    # Solo datasets con más de 100 imágenes
    if dataset_info.get('image_count', 0) < 100:
        return False
    
    # Solo datasets con anotaciones
    if not dataset_info.get('has_annotations', False):
        return False
    
    # Solo formatos YOLO o COCO
    if dataset_info.get('annotation_format') not in ['yolo', 'coco']:
        return False
    
    return True

# Aplicar filtro
analyzer = AdvancedDatasetAnalyzer("_dataSets")
analyzer.set_dataset_filter(custom_dataset_filter)
results = analyzer.analyze_all_datasets()
```

## 🚀 Ejemplos de Automatización

### Script de Análisis Completo
```python
#!/usr/bin/env python3
"""Script de análisis completo automatizado."""

import sys
from pathlib import Path
from Utils.advanced_analysis import analyze_dental_datasets
from Src.workflow_manager import DentalWorkflowManager

def main():
    # Configuración
    datasets_path = "_dataSets"
    output_path = "Dist/dental_ai"
    statistics_path = "StatisticsResults"
    
    print("🦷 Iniciando análisis completo...")
    
    # 1. Análisis de datasets
    print("📊 Fase 1: Análisis de datasets")
    analysis_results = analyze_dental_datasets(datasets_path, statistics_path)
    
    # 2. Crear estructura de trabajo
    print("🏗️ Fase 2: Creando estructura")
    workflow = DentalWorkflowManager(datasets_path, output_path)
    workflow.create_base_structure()
    
    # 3. Procesar datasets principales
    print("🔄 Fase 3: Procesando datasets")
    workflow.process_top_datasets(limit=5)
    
    # 4. Generar scripts
    print("📝 Fase 4: Generando scripts")
    scripts = workflow.generate_all_training_scripts()
    
    # 5. Crear documentación
    print("📋 Fase 5: Generando documentación")
    workflow.generate_comprehensive_docs()
    
    print("✅ Análisis completo terminado")
    print(f"📁 Resultados en: {output_path}")
    print(f"📊 Estadísticas en: {statistics_path}")

if __name__ == "__main__":
    main()
```

### Monitoreo Continuo
```python
# Script para monitoreo continuo de nuevos datasets
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DatasetMonitor(FileSystemEventHandler):
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def on_created(self, event):
        if event.is_directory:
            print(f"Nuevo dataset detectado: {event.src_path}")
            # Analizar automáticamente
            self.analyzer.analyze_new_dataset(event.src_path)

# Configurar monitor
analyzer = AdvancedDatasetAnalyzer("_dataSets")
monitor = DatasetMonitor(analyzer)

observer = Observer()
observer.schedule(monitor, "_dataSets", recursive=True)
observer.start()

print("🔍 Monitor de datasets iniciado...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    observer.join()
```

## 🎓 Ejemplos Educativos

### Tutorial Paso a Paso
```python
# Tutorial completo para principiantes
def tutorial_completo():
    """Tutorial paso a paso del sistema."""
    
    print("🎓 Tutorial: Análisis de Datasets Dentales")
    print("=" * 50)
    
    # Paso 1: Verificar datasets
    print("\n📋 Paso 1: Verificando datasets disponibles...")
    from pathlib import Path
    
    datasets_dir = Path("_dataSets")
    if not datasets_dir.exists():
        print("❌ Directorio _dataSets no encontrado")
        return
    
    dataset_types = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.startswith('_')]
    print(f"✅ Encontrados {len(dataset_types)} tipos de datasets")
    
    # Paso 2: Análisis básico
    print("\n📊 Paso 2: Ejecutando análisis básico...")
    from Utils.advanced_analysis import AdvancedDatasetAnalyzer
    
    analyzer = AdvancedDatasetAnalyzer("_dataSets", "StatisticsResults")
    results = analyzer.analyze_all_datasets()
    
    # Paso 3: Mostrar resultados
    print("\n📈 Paso 3: Resultados del análisis")
    if 'summary' in results:
        summary = results['summary']
        print(f"   📁 Datasets totales: {summary.get('total_datasets', 0)}")
        print(f"   🖼️ Imágenes totales: {summary.get('total_images', 0):,}")
        print(f"   🏷️ Categorías únicas: {len(summary.get('top_categories', {}))}")
    
    # Paso 4: Crear visualizaciones
    print("\n🎨 Paso 4: Generando visualizaciones...")
    from Utils.advanced_analysis import DentalDatasetStatisticsViewer
    
    viewer = DentalDatasetStatisticsViewer(output_dir="StatisticsResults")
    viewer.generate_all_reports()
    
    print("\n✅ Tutorial completado!")
    print("📁 Revisa StatisticsResults/ para ver los resultados")

# Ejecutar tutorial
tutorial_completo()
```

## 📞 Soporte y Recursos

### Depuración de Problemas
```python
# Función de diagnóstico
def diagnosticar_sistema():
    """Diagnostica problemas comunes del sistema."""
    import sys
    from pathlib import Path
    
    print("🔧 Diagnóstico del Sistema")
    print("=" * 30)
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar dependencias
    dependencias = ['numpy', 'pandas', 'opencv-python', 'matplotlib', 'seaborn']
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"✅ {dep}: Instalado")
        except ImportError:
            print(f"❌ {dep}: No instalado")
    
    # Verificar estructura
    dirs_requeridos = ['Src', 'Utils', 'StatisticsResults', 'Wiki']
    for dir_name in dirs_requeridos:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/: Existe")
        else:
            print(f"❌ {dir_name}/: No existe")
    
    # Verificar datasets
    datasets_path = Path("_dataSets")
    if datasets_path.exists():
        dataset_count = len(list(datasets_path.rglob("*.jpg"))) + len(list(datasets_path.rglob("*.png")))
        print(f"✅ _dataSets/: {dataset_count} imágenes encontradas")
    else:
        print("❌ _dataSets/: No existe")

# Ejecutar diagnóstico
diagnosticar_sistema()
```

### Logs Detallados
```python
import logging

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dental_workflow.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DentalWorkflow')

# Usar en el código
logger.info("Iniciando análisis de datasets")
logger.debug(f"Procesando directorio: {dataset_path}")
logger.warning(f"Dataset con pocas imágenes: {dataset_name}")
logger.error(f"Error procesando {file_path}: {error}")
```

---

*Para más ejemplos y documentación, consulta los archivos en la carpeta Wiki/.*
