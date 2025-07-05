# 🦷 GUÍA DE USO DEL WORKFLOW MANAGER

## 📋 Descripción General
El `DataWorkflowManager.py` implementa una estrategia completa para unificar, balancear y preparar datasets dentales para entrenamiento de modelos de IA.

## 🎯 Objetivos del Workflow
- **Unificación**: Convierte todos los datasets a formatos estándar
- **Balanceo**: Identifica y corrige desbalances en las clases
- **Optimización**: Maximiza el rendimiento mediante fusión inteligente
- **Estandarización**: Normaliza resoluciones y anotaciones

## 🔄 Fases del Workflow

### 1. 🧱 Unificación y Preprocesamiento
```
✅ Detección → YOLO format (.txt + data.yaml)
✅ Segmentación → COCO format (.json)  
✅ Clasificación → Estructura de carpetas
✅ Normalización de resoluciones (640x640, 1024x1024)
✅ Unificación de nombres de clases inconsistentes
```

### 2. 📊 Análisis Estadístico de Clases
```
🔍 Cuenta muestras por clase
⚖️ Identifica desbalances críticos
📈 Calcula factores de augmentación necesarios
🎯 Propone estrategias de mejora
```

### 3. 🔄 Fusión de Datasets Similares
```
🔗 Agrupa datasets compatibles:
   • dental_detection_panoramic
   • dental_detection_periapical  
   • dental_segmentation_coco
   • dental_classification

💡 Genera data.yaml unificado
🏷️ Mapea clases a identificadores únicos
```

### 4. 🧪 División Train/Val/Test
```
📊 Split estratificado: 70/20/10
👥 Evita mezclar datos del mismo paciente
🎲 Semilla aleatoria reproducible
✅ Validación de distribución balanceada
```

### 5. ⚙️ Recomendaciones de Entrenamiento
```
🤖 Arquitecturas recomendadas por tamaño:
   • <5K imágenes: YOLOv8n, Mask R-CNN ResNet50
   • <20K imágenes: YOLOv8s, Mask R-CNN ResNet101  
   • >20K imágenes: YOLOv8m/l, ensemble models

⚡ Configuración optimizada:
   • Learning rates adaptativos
   • Batch sizes según hardware
   • Epochs basados en convergencia esperada
```

## 🏷️ Clases Unificadas Soportadas

```yaml
caries: [caries, Caries, CARIES, cavity, decay, Q1_Caries, Q2_Caries, Q3_Caries, Q4_Caries]
tooth: [tooth, teeth, Tooth, TOOTH, diente, molar, premolar, canine, incisor]
filling: [filling, Filling, Fillings, FILLING, restoration, RESTORATION]
crown: [crown, Crown, CROWN, CROWN AND BRIDGE]
implant: [implant, Implant, IMPLANT]
root_canal: [Root Canal Treatment, ROOT CANAL TREATED TOOTH, root canal]
bone_loss: [Bone Loss, BONE LOSS, VERTICAL BONE LOSS]
impacted: [impacted, Impacted, IMPACTED TOOTH, Q1_Impacted, Q2_Impacted, Q3_Impacted, Q4_Impacted]
periapical_lesion: [Periapical lesion, Q1_Periapical_Lesion, Q2_Periapical_Lesion, Q3_Periapical_Lesion, Q4_Periapical_Lesion]
maxillary_sinus: [maxillary sinus, MAXILLARY SINUS, MAXILLARY  SINUS]
mandible: [Mandible, mandible, RAMUS OF MANDIBLE, INFERIOR BORDER OF MANDIBLE]
maxilla: [Maxilla, maxilla]
```

## 🚀 Ejemplos de Uso

### Ejecutar Workflow Completo
```bash
cd /Volumes/3TB/Ai/XRAY
python DataWorkflowManager.py
# Seleccionar opción 6: "Ejecutar workflow completo"
```

### Crear Dataset YOLO Unificado
```python
from DataWorkflowManager import DentalDataWorkflowManager

manager = DentalDataWorkflowManager(
    base_path="/Volumes/3TB/Ai/XRAY/_dataSets",
    output_path="/Volumes/3TB/Ai/XRAY/processed_datasets"
)

# Datasets a fusionar
datasets_to_merge = [
    "dental_diseases_yolo",
    "Dental Diseases", 
    "dental_datasetv6",
    "Dental 3"
]

# Crear dataset unificado
unified_path = manager.create_unified_yolo_dataset(
    data=manager.load_analysis_data(),
    datasets_to_merge=datasets_to_merge,
    output_name="unified_dental_panoramic_detection"
)
```

### Analizar Balance de Clases
```python
manager = DentalDataWorkflowManager(base_path)
data = manager.load_analysis_data()
class_dist, total_samples = manager.analyze_class_distribution(data)
imbalanced = manager.identify_imbalanced_classes(total_samples)
```

## 📊 Archivos de Salida

### Estructura de Dataset Unificado
```
unified_dental_detection/
├── data.yaml              # Configuración YOLO
├── classes.txt            # Lista de clases
├── train/
│   ├── images/            # Imágenes de entrenamiento
│   └── labels/            # Anotaciones YOLO (.txt)
├── val/
│   ├── images/            # Imágenes de validación  
│   └── labels/            # Anotaciones YOLO (.txt)
└── test/
    ├── images/            # Imágenes de prueba
    └── labels/            # Anotaciones YOLO (.txt)
```

### Archivos de Análisis
```
processed_datasets/
├── workflow_config.json   # Configuración completa
├── workflow_analysis.png  # Visualización del análisis
├── class_distribution.json # Distribución detallada
└── training_recommendations.json # Recomendaciones
```

## 🎯 Estrategias de Augmentación

### Para Clases Muy Desbalanceadas (< 30% del promedio)
```yaml
techniques: [rotation, brightness, contrast, noise, flip_horizontal]
factor: 3-4x
priority: high
```

### Para Clases Moderadamente Desbalanceadas (30-70% del promedio)  
```yaml
techniques: [rotation, brightness]
factor: 2x
priority: medium
```

## 🤖 Recomendaciones de Entrenamiento

### Dataset Pequeño (< 5,000 imágenes)
```yaml
architecture: YOLOv8n
epochs: 100
batch_size: 16
learning_rate: 0.01
augmentation: heavy
strategy: transfer_learning
```

### Dataset Mediano (5,000 - 20,000 imágenes)
```yaml
architecture: YOLOv8s
epochs: 150  
batch_size: 32
learning_rate: 0.01
augmentation: medium
strategy: fine_tuning
```

### Dataset Grande (> 20,000 imágenes)
```yaml
architecture: YOLOv8m
epochs: 200
batch_size: 32-64
learning_rate: 0.005
augmentation: light
strategy: from_scratch + ensemble
```

## 💻 Requisitos de Hardware

### Configuración Mínima
- GPU: RTX 3060 (8GB VRAM)
- RAM: 16GB
- Storage: 100GB SSD
- Tiempo estimado: 4-12 horas

### Configuración Recomendada  
- GPU: RTX 4090 (24GB VRAM)
- RAM: 32GB+
- Storage: 500GB+ NVMe SSD
- Tiempo estimado: 2-7 días

## 🔧 Personalización

### Modificar Ratios de División
```python
manager.workflow_config['train_ratio'] = 0.8
manager.workflow_config['val_ratio'] = 0.15  
manager.workflow_config['test_ratio'] = 0.05
```

### Añadir Nuevas Clases Unificadas
```python
manager.unified_classes['nueva_clase'] = ['variante1', 'variante2', 'VARIANTE3']
```

### Cambiar Resoluciones Estándar
```python
manager.standard_resolutions['yolo'] = (512, 512)
manager.standard_resolutions['coco'] = (800, 800)
```

## ⚠️ Consideraciones Importantes

1. **Backup**: Siempre haz backup de datasets originales antes de procesarlos
2. **Memoria**: Datasets grandes requieren RAM suficiente para procesamiento
3. **Validación**: Revisa manualmente muestras del dataset unificado
4. **Consistencia**: Verifica que las anotaciones se conserven correctamente
5. **Derechos**: Respeta licencias y términos de uso de datasets públicos

## 🆘 Solución de Problemas

### Error: "No se encontró dental_dataset_analysis.json"
```bash
# Ejecutar primero el análisis de datasets
python script.py
```

### Error de memoria durante procesamiento
```python
# Procesar en lotes más pequeños
manager.workflow_config['batch_processing'] = True
manager.workflow_config['batch_size'] = 1000
```

### Clases no reconocidas
```python
# Añadir mapeo manual
manager.unified_classes['clase_personalizada'] = ['nombre_original']
```
