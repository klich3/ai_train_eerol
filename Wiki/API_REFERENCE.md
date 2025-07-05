# 📖 API Reference - Dental AI Workflow Manager

## 🏗️ Módulos Principales

### Src.workflow_manager

#### `DentalWorkflowManager`
Gestor principal del workflow para datasets dentales.

```python
class DentalWorkflowManager:
    def __init__(self, source_dir: str, output_dir: str, project_name: str = "dental_ai"):
        """
        Inicializa el gestor de workflow.
        
        Args:
            source_dir: Directorio fuente con datasets
            output_dir: Directorio de salida
            project_name: Nombre del proyecto
        """
```

**Métodos principales:**

##### `create_base_structure() -> Path`
Crea la estructura base del proyecto.
```python
# Retorna la ruta de la estructura creada
structure_path = workflow.create_base_structure()
```

##### `find_compatible_datasets(architecture: str) -> List[str]`
Encuentra datasets compatibles con una arquitectura específica.
```python
# Buscar datasets YOLO
yolo_datasets = workflow.find_compatible_datasets('yolo')
```

##### `process_dataset(dataset_path: str, architecture: str, output_path: str)`
Procesa un dataset para una arquitectura específica.
```python
workflow.process_dataset(
    dataset_path="datasets/dental_yolo",
    architecture="yolo", 
    output_path="output/yolo"
)
```

##### `generate_training_scripts(architectures: List[str]) -> Dict[str, str]`
Genera scripts de entrenamiento para arquitecturas especificadas.
```python
scripts = workflow.generate_training_scripts(['yolo', 'coco', 'unet'])
# Retorna: {'yolo': 'path/to/yolo_train.py', ...}
```

### Src.data_analyzer

#### `DatasetAnalyzer`
Analizador de estructura y contenido de datasets.

```python
class DatasetAnalyzer:
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Ruta base donde están los datasets
        """
```

**Métodos principales:**

##### `analyze_dataset_structure(dataset_path: str) -> Dict[str, Any]`
Analiza la estructura de un dataset.
```python
structure_info = analyzer.analyze_dataset_structure("datasets/dental_dataset")
# Retorna información sobre directorios, archivos, formato detectado
```

##### `count_images_and_annotations(dataset_path: str) -> Tuple[int, int]`
Cuenta imágenes y anotaciones en un dataset.
```python
image_count, annotation_count = analyzer.count_images_and_annotations(dataset_path)
```

##### `detect_format(dataset_path: str) -> str`
Detecta automáticamente el formato del dataset.
```python
format_type = analyzer.detect_format(dataset_path)
# Retorna: 'yolo', 'coco', 'pascal_voc', 'classification', etc.
```

### Src.data_processor

#### `BaseProcessor`
Clase base para procesadores de datos.

#### `YOLOProcessor(BaseProcessor)`
Procesador especializado para formato YOLO.

```python
class YOLOProcessor(BaseProcessor):
    def create_yolo_dataset(self, source_dirs: List[str], output_dir: str, 
                          class_mapping: Dict[str, int] = None) -> str:
        """
        Crea un dataset YOLO unificado.
        
        Args:
            source_dirs: Directorios fuente
            output_dir: Directorio de salida
            class_mapping: Mapeo de clases a IDs
        
        Returns:
            Ruta del dataset creado
        """
```

##### `generate_data_yaml(dataset_info: Dict, output_path: str)`
Genera archivo data.yaml para YOLO.
```python
processor.generate_data_yaml({
    'train_path': 'datasets/train',
    'val_path': 'datasets/val',
    'classes': ['tooth', 'caries']
}, 'output/data.yaml')
```

#### `COCOProcessor(BaseProcessor)`
Procesador para formato COCO.

```python
class COCOProcessor(BaseProcessor):
    def merge_coco_annotations(self, annotation_files: List[str], 
                             output_file: str) -> str:
        """
        Fusiona múltiples archivos de anotaciones COCO.
        
        Args:
            annotation_files: Lista de archivos JSON COCO
            output_file: Archivo de salida fusionado
        
        Returns:
            Ruta del archivo fusionado
        """
```

### Src.structure_generator

#### `ProjectStructureGenerator`
Generador de estructuras de proyecto.

```python
class ProjectStructureGenerator:
    def __init__(self, base_output_dir: str):
        """
        Args:
            base_output_dir: Directorio base donde crear la estructura
        """
```

##### `create_architecture_structure(architecture: str, config: Dict = None) -> str`
Crea estructura específica para una arquitectura.
```python
yolo_structure = generator.create_architecture_structure('yolo', {
    'include_training_scripts': True,
    'include_validation_scripts': True,
    'create_sample_configs': True
})
```

## 🔧 Utilidades (Utils)

### Utils.advanced_analysis

#### `AdvancedDatasetAnalyzer`
Analizador avanzado con análisis de calidad de imágenes.

```python
class AdvancedDatasetAnalyzer:
    def __init__(self, base_path: str, output_dir: str = "StatisticsResults"):
        """
        Args:
            base_path: Directorio de datasets
            output_dir: Directorio para resultados
        """
```

##### `analyze_all_datasets() -> Dict[str, Any]`
Ejecuta análisis completo de todos los datasets.
```python
results = analyzer.analyze_all_datasets()
# Retorna diccionario con análisis completo y resumen
```

##### `analyze_image_quality(image_path: str) -> Dict[str, float]`
Analiza la calidad de una imagen específica.
```python
quality_metrics = analyzer.analyze_image_quality("dataset/image.jpg")
# Retorna: {'contrast': 0.8, 'sharpness': 0.7, 'quality_score': 0.75}
```

#### `DentalDatasetStatisticsViewer`
Generador de visualizaciones y reportes.

```python
class DentalDatasetStatisticsViewer:
    def __init__(self, json_file: str = None, output_dir: str = "StatisticsResults"):
        """
        Args:
            json_file: Archivo JSON con datos de análisis
            output_dir: Directorio para guardar visualizaciones
        """
```

##### `generate_all_reports()`
Genera todos los reportes y visualizaciones disponibles.
```python
viewer.generate_all_reports()
# Crea gráficos PNG, tablas CSV, dashboard HTML
```

##### `create_custom_visualization(chart_type: str, data: Dict, output_file: str)`
Crea visualización personalizada.
```python
viewer.create_custom_visualization(
    chart_type='bar',
    data={'categories': ['caries', 'tooth'], 'counts': [150, 300]},
    output_file='custom_chart.png'
)
```

### Utils.visualization

#### `DatasetVisualizer`
Visualizador especializado para datasets dentales.

```python
class DatasetVisualizer:
    def __init__(self, data_source: str):
        """
        Args:
            data_source: Ruta al archivo JSON de análisis
        """
```

##### `create_distribution_chart(chart_type: str = 'pie') -> str`
Crea gráfico de distribución.
```python
chart_path = visualizer.create_distribution_chart('bar')
```

##### `create_quality_heatmap(datasets: List[str] = None) -> str`
Crea mapa de calor de calidad de datasets.
```python
heatmap_path = visualizer.create_quality_heatmap(['dataset1', 'dataset2'])
```

### Utils.data_augmentation

#### `DataBalancer`
Balanceador de datasets desbalanceados.

```python
class DataBalancer:
    def __init__(self, target_samples: int = 1000):
        """
        Args:
            target_samples: Número objetivo de muestras por clase
        """
```

##### `balance_dataset(dataset_path: str, output_path: str, techniques: List[str])`
Balancea un dataset usando técnicas de augmentación.
```python
balancer.balance_dataset(
    dataset_path="imbalanced_dataset",
    output_path="balanced_dataset", 
    techniques=['rotation', 'flip', 'brightness', 'noise']
)
```

#### `QualityChecker`
Verificador de calidad de imágenes.

```python
class QualityChecker:
    def check_image_quality(self, image_path: str, 
                          min_resolution: Tuple[int, int] = (224, 224),
                          min_quality_score: float = 0.5) -> bool:
        """
        Verifica si una imagen cumple los criterios de calidad.
        
        Args:
            image_path: Ruta a la imagen
            min_resolution: Resolución mínima requerida
            min_quality_score: Puntuación mínima de calidad
        
        Returns:
            True si la imagen cumple los criterios
        """
```

### Utils.dental_format_converter

#### `DentalFormatConverter`
Convertidor entre formatos de anotación.

```python
class DentalFormatConverter:
    def coco_to_yolo(self, coco_file: str, output_dir: str, 
                     image_dir: str = None) -> str:
        """
        Convierte anotaciones COCO a formato YOLO.
        
        Args:
            coco_file: Archivo JSON COCO
            output_dir: Directorio de salida YOLO
            image_dir: Directorio de imágenes (opcional)
        
        Returns:
            Directorio YOLO creado
        """
```

##### `yolo_to_coco(yolo_dir: str, output_file: str, categories: List[str])`
Convierte formato YOLO a COCO.
```python
converter.yolo_to_coco(
    yolo_dir="yolo_dataset",
    output_file="coco_annotations.json",
    categories=['tooth', 'caries', 'implant']
)
```

##### `pascal_voc_to_yolo(xml_files: List[str], output_dir: str)`
Convierte Pascal VOC a YOLO.
```python
converter.pascal_voc_to_yolo(
    xml_files=glob.glob("annotations/*.xml"),
    output_dir="yolo_converted"
)
```

## 🚀 Funciones de Conveniencia

### Función Principal de Análisis
```python
from Utils.advanced_analysis import analyze_dental_datasets

def analyze_dental_datasets(datasets_path: str, 
                          output_dir: str = "StatisticsResults",
                          generate_visuals: bool = True) -> Dict[str, Any]:
    """
    Función principal para análisis completo de datasets dentales.
    
    Args:
        datasets_path: Ruta al directorio con datasets
        output_dir: Directorio para guardar resultados
        generate_visuals: Si generar visualizaciones automáticamente
    
    Returns:
        Diccionario con resultados completos del análisis
    """
```

### Configuración Global
```python
# Configurar el sistema globalmente
from Src.config import DentalConfig

config = DentalConfig({
    'default_image_size': (640, 640),
    'supported_formats': ['.jpg', '.png', '.tiff'],
    'quality_threshold': 0.7,
    'batch_size': 32,
    'num_workers': 4
})

# Aplicar configuración
DentalConfig.set_global_config(config)
```

## 📊 Estructuras de Datos

### Resultado de Análisis de Dataset
```python
dataset_analysis = {
    'name': str,                    # Nombre del dataset
    'path': str,                    # Ruta del dataset
    'format_type': str,             # Tipo de formato detectado
    'image_count': int,             # Número de imágenes
    'annotation_count': int,        # Número de anotaciones
    'categories': List[str],        # Categorías encontradas
    'class_distribution': Dict[str, int],  # Distribución de clases
    'size_distribution': {          # Distribución por tamaño
        'small': int,
        'medium': int, 
        'large': int
    },
    'quality_assessment': {         # Evaluación de calidad
        'good': int,
        'fair': int,
        'poor': int
    },
    'has_annotations': bool,        # Si tiene anotaciones
    'annotation_format': str        # Formato de anotaciones
}
```

### Resumen General
```python
summary = {
    'total_dataset_types': int,     # Tipos de datasets
    'total_datasets': int,          # Total de datasets
    'total_images': int,            # Total de imágenes
    'format_distribution': Dict[str, int],  # Distribución por formato
    'top_categories': Dict[str, int],       # Categorías más frecuentes
    'quality_overview': Dict[str, int],     # Resumen de calidad
    'size_overview': Dict[str, int]         # Resumen de tamaños
}
```

### Configuración de Proyecto
```python
project_config = {
    'project_name': str,
    'architectures': List[str],     # ['yolo', 'coco', 'unet']
    'target_classes': List[str],    # Clases objetivo
    'image_size': Tuple[int, int],  # Tamaño objetivo de imágenes
    'train_split': float,           # Proporción de entrenamiento
    'val_split': float,             # Proporción de validación
    'test_split': float,            # Proporción de prueba
    'augmentation_config': {        # Configuración de augmentación
        'enabled': bool,
        'techniques': List[str],
        'intensity': float
    }
}
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Configurar directorios por defecto
export DENTAL_DATASETS_PATH="_dataSets"
export DENTAL_OUTPUT_PATH="Dist/dental_ai"
export DENTAL_STATISTICS_PATH="StatisticsResults"

# Configurar comportamiento
export DENTAL_AUTO_ANALYZE=true
export DENTAL_GENERATE_VISUALS=true
export DENTAL_VERBOSE_LOGGING=true
```

### Archivos de Configuración
```yaml
# dental_config.yaml
datasets:
  base_path: "_dataSets"
  supported_formats: [".jpg", ".png", ".tiff"]
  min_image_size: [224, 224]
  
analysis:
  quality_threshold: 0.7
  enable_image_analysis: true
  enable_annotation_analysis: true
  
output:
  base_path: "Dist/dental_ai"
  statistics_path: "StatisticsResults"
  create_backups: true
  
processing:
  batch_size: 32
  num_workers: 4
  use_gpu: true
```

## ⚠️ Manejo de Errores

### Excepciones Personalizadas
```python
class DentalWorkflowError(Exception):
    """Excepción base del workflow dental."""
    pass

class DatasetNotFoundError(DentalWorkflowError):
    """Dataset no encontrado."""
    pass

class UnsupportedFormatError(DentalWorkflowError):
    """Formato no soportado."""
    pass

class InvalidAnnotationError(DentalWorkflowError):
    """Anotación inválida."""
    pass
```

### Manejo de Errores
```python
try:
    results = analyzer.analyze_all_datasets()
except DatasetNotFoundError as e:
    logger.error(f"Dataset no encontrado: {e}")
except UnsupportedFormatError as e:
    logger.warning(f"Formato no soportado: {e}")
except Exception as e:
    logger.error(f"Error inesperado: {e}")
    # Continuar con el siguiente dataset
```

## 📈 Métricas y Logging

### Métricas Disponibles
```python
metrics = {
    'processing_time': float,       # Tiempo de procesamiento (segundos)
    'memory_usage': float,          # Uso de memoria (MB)
    'success_rate': float,          # Tasa de éxito (0.0-1.0)
    'error_count': int,             # Número de errores
    'datasets_processed': int,      # Datasets procesados
    'images_analyzed': int          # Imágenes analizadas
}
```

### Configuración de Logging
```python
import logging

# Configurar logger personalizado
logger = logging.getLogger('DentalWorkflow')
logger.setLevel(logging.INFO)

# Handler para archivo
file_handler = logging.FileHandler('dental_workflow.log')
file_handler.setLevel(logging.DEBUG)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

---

*Esta documentación corresponde a la versión 2.0 del Dental AI Workflow Manager.*
