# 🦷 Guía Completa del Workflow Manager Dental-AI

## 📋 Resumen

El **DataWorkflowManager** es un sistema robusto y seguro para la gestión, análisis, fusión y preparación de datasets dentales para IA. Implementa una arquitectura de seguridad que **NUNCA modifica los datos originales** y crea una estructura organizacional completa lista para entrenamiento y despliegue.

## 🛡️ Características de Seguridad

### Protección de Datos Originales
- ✅ **SOLO LECTURA**: Los datos originales nunca se modifican
- ✅ **Verificación de integridad**: Hash MD5 para archivos copiados
- ✅ **Permisos restringidos**: Archivos copiados como solo lectura
- ✅ **Trazabilidad completa**: Logs detallados de todas las operaciones
- ✅ **Backup automático**: Metadatos preservados

### Estructura de Seguridad
```
_dataSets/           # 🔒 DATOS ORIGINALES (PROTEGIDOS)
dental-ai/           # 📁 DATOS PROCESADOS (NUEVA CREACIÓN)
├── datasets/        # Datasets fusionados y listos
├── models/          # Modelos entrenados
├── training/        # Scripts y configuraciones
├── api/             # API REST
└── docs/            # Documentación
```

## 🚀 Inicio Rápido

### 1. Requisitos Previos
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar análisis inicial (si no está hecho)
python AnalizeDataSets.py
```

### 2. Ejecutar Workflow Manager
```bash
python DataWorkflowManager.py
```

### 3. Menú Principal
```
🎯 ¿Qué operación quieres realizar?
1. 🔍 Analizar distribución de clases
2. 🔗 Recomendar fusión de datasets
3. 🧱 Crear dataset YOLO (detección) unificado
4. 🎨 Crear dataset COCO (segmentación) unificado
5. 📂 Crear dataset de clasificación unificado
6. ⚖️ Analizar balance de clases
7. 🎯 Generar estrategia de entrenamiento
8. 🚀 Ejecutar workflow completo
9. 📊 Ver configuración de seguridad
10. 🛠️ Crear estructura dental-ai completa
11. 🌐 Crear plantilla de API
12. ❌ Salir
```

## 📊 Funcionalidades Principales

### 1. Análisis de Datasets 🔍
- **Distribución de clases**: Análisis estadístico por formato (YOLO, COCO, etc.)
- **Unificación de nombres**: Mapeo automático de clases similares
- **Detección de desbalances**: Identificación de clases subrepresentadas
- **Visualizaciones**: Gráficos de distribución y balance

### 2. Fusión de Datasets 🔗
- **Recomendaciones automáticas**: Agrupación inteligente por tipo y contenido
- **Compatibilidad de formatos**: YOLO, COCO, clasificación por carpetas
- **Preservación de metadatos**: Trazabilidad de datasets fuente

### 3. Creación de Datasets Unificados 🧱

#### Dataset YOLO (Detección)
```bash
# Opción 3 en el menú
Datasets a fusionar: dataset1, dataset2, dataset3
Nombre de salida: panoramic_detection
```

**Resultado:**
```
dental-ai/datasets/detection_combined/panoramic_detection/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── classes.txt
```

**Archivos generados:**
- `data.yaml`: Configuración para YOLO
- `classes.txt`: Lista de clases unificadas
- Scripts de entrenamiento específicos
- Logs de procesamiento completos

#### Dataset COCO (Segmentación)
```bash
# Opción 4 en el menú
Datasets a fusionar: coco_dataset1, coco_dataset2
Nombre de salida: dental_segmentation
```

**Resultado:**
```
dental-ai/datasets/segmentation_coco/dental_segmentation/
├── train/
│   ├── images...
│   └── annotations.json
├── val/
│   ├── images...
│   └── annotations.json
└── test/
    ├── images...
    └── annotations.json
```

#### Dataset de Clasificación
```bash
# Opción 5 en el menú
Datasets a fusionar: classification1, classification2
Nombre de salida: dental_pathology_classification
```

**Resultado:**
```
dental-ai/datasets/classification/dental_pathology_classification/
├── train/
│   ├── normal/
│   ├── caries/
│   └── filling/
├── val/
│   ├── normal/
│   ├── caries/
│   └── filling/
├── test/
│   ├── normal/
│   ├── caries/
│   └── filling/
├── classes.txt
└── metadata.json
```

### 4. Scripts de Entrenamiento Automáticos 🏋️

Para cada dataset creado, se generan scripts de entrenamiento específicos:

#### YOLO (Detección)
```bash
cd dental-ai/training
bash train_panoramic_detection.sh
```

#### Segmentación (Detectron2)
```bash
cd dental-ai/training
python train_seg_dental_segmentation.py
```

#### Clasificación (PyTorch)
```bash
cd dental-ai/training
python train_cls_dental_pathology_classification.py
```

### 5. API de Inferencia 🌐

```bash
# Crear plantilla de API
# Opción 11 en el menú

cd dental-ai/api
pip install -r requirements.txt
python main.py

# Navegar a: http://localhost:8000/docs
```

**Endpoints disponibles:**
- `POST /predict/detection`: Detección YOLO
- `POST /predict/classification`: Clasificación CNN
- `GET /models`: Listar modelos disponibles
- `GET /health`: Estado del sistema

## 🎯 Estrategias de Entrenamiento

### Balance de Clases ⚖️
El sistema identifica automáticamente clases desbalanceadas y sugiere:
- **Augmentación selectiva**: Factor de multiplicación por clase
- **Técnicas específicas**: Rotación, brillo, contraste, ruido
- **Priorización**: Alta, media, baja según el desbalance

### División Estratificada 📊
- **Train**: 70% (por defecto)
- **Validation**: 20%
- **Test**: 10%
- **Estratificación por clase**: Mantiene proporción en todos los splits

### Recomendaciones de Arquitectura 🏗️
Basadas en el tamaño del dataset:
- **< 5K muestras**: YOLOv8n, ResNet50, entrenamiento intensivo
- **5K-20K muestras**: YOLOv8s, ResNet101, augmentación media
- **> 20K muestras**: YOLOv8m/l, arquitecturas más grandes

## 📈 Monitoreo y Logging

### Logs de Procesamiento
```
dental-ai/training/logs/[dataset_name]_creation/
├── [dataset_name]_processing.log    # Log detallado
├── [dataset_name]_stats.json        # Estadísticas
└── safety_log.json                  # Log de seguridad
```

### Métricas Registradas
- **Datasets procesados**: Número y nombres
- **Imágenes copiadas**: Total y por split
- **Errores encontrados**: Detalles y soluciones
- **Integridad verificada**: Hash y validaciones
- **Tiempo de procesamiento**: Timestamps completos

## 🔧 Configuración Avanzada

### Personalización de Clases Unificadas
Editar en `DataWorkflowManager.py`:
```python
self.unified_classes = {
    'caries': ['caries', 'Caries', 'CARIES', 'cavity', 'decay'],
    'tooth': ['tooth', 'teeth', 'Tooth', 'diente'],
    # Agregar más mapeos...
}
```

### Configuración de Workflow
```python
self.workflow_config = {
    'train_ratio': 0.7,      # Modificar splits
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'min_samples_per_class': 10,
    'max_augmentation_factor': 5,
    'class_balance_threshold': 0.1
}
```

### Configuración de Seguridad
```python
self.safety_config = {
    'backup_enabled': True,           # Backups automáticos
    'read_only_source': True,         # Solo lectura en origen
    'verify_copy': True,              # Verificar integridad
    'preserve_original_structure': True
}
```

## 🚨 Solución de Problemas

### Error: Dataset no encontrado
```bash
❌ Dataset no encontrado: /path/to/dataset
```
**Solución**: Verificar que el dataset existe en `_dataSets/` y que el análisis previo lo detectó.

### Error: Sin permisos de lectura
```bash
❌ Sin permisos de lectura: /path/to/file
```
**Solución**: Verificar permisos del sistema de archivos:
```bash
chmod -R +r _dataSets/
```

### Error: Verificación de integridad falló
```bash
❌ Verificación de integridad falló: source -> destination
```
**Solución**: Verificar espacio en disco y permisos de escritura en `dental-ai/`.

### Datasets vacíos después de fusión
**Causas comunes:**
- Nombres de datasets incorrectos
- Estructura de carpetas no reconocida
- Archivos de anotaciones corruptos

**Solución**: Revisar logs en `dental-ai/training/logs/` para detalles específicos.

## 📚 Estructura de Archivos Generados

### Dataset YOLO Completo
```
dental-ai/datasets/detection_combined/[nombre]/
├── train/
│   ├── images/          # Imágenes de entrenamiento
│   └── labels/          # Anotaciones YOLO (.txt)
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml            # Configuración YOLO
├── classes.txt          # Lista de clases
└── README.md            # Documentación del dataset
```

### Scripts de Entrenamiento
```
dental-ai/training/
├── scripts/
│   ├── train_[nombre].sh           # Script YOLO
│   ├── train_seg_[nombre].py       # Script segmentación
│   └── train_cls_[nombre].py       # Script clasificación
├── configs/
│   └── [configuraciones específicas]
└── logs/
    ├── [nombre]_creation/          # Logs de creación
    └── [nombre]_training/          # Logs de entrenamiento
```

### API Complete
```
dental-ai/api/
├── main.py              # API principal FastAPI
├── requirements.txt     # Dependencias específicas
├── models/              # Directorio para modelos
└── utils/               # Utilidades
```

## 🎯 Mejores Prácticas

### 1. Preparación de Datos
- **Ejecutar análisis completo** antes de fusionar datasets
- **Verificar estructura** de carpetas y anotaciones
- **Revisar nombres de clases** para mapeo correcto

### 2. Fusión de Datasets
- **Empezar con pocos datasets** para validar el proceso
- **Verificar compatibilidad** de formatos antes de fusionar
- **Revisar logs** después de cada operación

### 3. Entrenamiento
- **Usar scripts generados** como punto de partida
- **Monitorear métricas** durante el entrenamiento
- **Guardar checkpoints** regulares

### 4. Despliegue
- **Probar API localmente** antes de producción
- **Validar modelos** con datos de test
- **Implementar monitoreo** de performance

## 🔮 Extensiones Futuras

### Formatos Adicionales
- **Pascal VOC**: Conversión automática
- **TFRecord**: Para TensorFlow
- **Darknet**: Para versiones específicas de YOLO

### Augmentación Avanzada
- **GAN-based augmentation**: Para clases muy pequeñas
- **Medical-specific transforms**: Transformaciones específicas para rayos X
- **Synthetic data generation**: Generación de datos sintéticos

### Integración MLOps
- **MLflow**: Tracking de experimentos
- **DVC**: Versionado de datos
- **Kubernetes**: Despliegue escalable

## 📞 Soporte

Para reportar problemas o sugerir mejoras:
1. Revisar logs en `dental-ai/training/logs/`
2. Verificar configuración de seguridad
3. Consultar esta documentación
4. Crear issue con detalles específicos

---

**🛡️ Recuerda: Este sistema garantiza que tus datos originales NUNCA se modificarán. Todas las operaciones son seguras y reversibles.**
