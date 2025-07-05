# 🧠 Smart Dental AI Workflow Manager v3.0

## 📋 Descripción

Sistema inteligente y modular para análisis, conversión y preparación de datasets dentales con menú interactivo, análisis automático de categorías y verificación de calidad.

## 🎯 Características Principales

### 🔍 Análisis Inteligente
- **Escaneo automático** de estructuras de datasets
- **Detección de formatos** (YOLO, COCO, U-Net, Clasificación)
- **Análisis de categorías** con mapeo inteligente
- **Métricas de calidad** automáticas
- **Detección de patrones** en nombres de clases

### 📊 Gestión de Categorías
- **Mapeo unificado** de clases dentales
- **Detección de patrones** regex para categorización
- **Análisis de distribución** de clases
- **Recomendaciones automáticas** de mejora
- **Reportes detallados** en JSON y Markdown

### 🔄 Conversión de Formatos
- **YOLO** para detección de objetos
- **COCO** para detección y segmentación
- **U-Net** para segmentación médica
- **Clasificación** por directorios
- **Conversión múltiple** automática

### ⚖️ Balanceado de Datos
- **Análisis de distribución** actual
- **Balanceado automático** inteligente
- **Configuración personalizada** por categoría
- **Augmentación de datos** dirigida
- **Métricas de balance** en tiempo real

### ✅ Verificación y Validación
- **Verificación de estructura** de directorios
- **Integridad de datos** (imágenes y anotaciones)
- **Distribución de clases** validada
- **Reportes de validación** detallados
- **Recomendaciones de corrección**

## 🚀 Modos de Uso

### 1. 🎮 Modo Interactivo (Recomendado)
```bash
python smart_dental_workflow.py
```

**Menú principal:**
- 🔍 Escanear y analizar datasets
- 📊 Ver categorías disponibles  
- 📦 Seleccionar datasets
- 🔄 Convertir formatos
- ⚖️ Balancear datasets
- ✅ Verificar y validar
- 📝 Generar scripts de entrenamiento
- 🚀 Workflow completo
- 📋 Reporte de análisis

### 2. 🤖 Modo Automático
```bash
python smart_dental_workflow.py --mode auto
```
Ejecuta todo el workflow sin intervención manual.

### 3. ⚡ Modo Análisis Rápido
```bash
python smart_dental_workflow.py --mode analysis
```
Solo escanea y muestra un resumen de categorías.

## 📁 Estructura de Salida

```
Dist/dental_ai/
├── 📊 analysis/
│   ├── dataset_analysis_YYYYMMDD_HHMMSS.json
│   └── category_analysis_report.md
├── 📦 datasets/
│   ├── yolo/
│   │   ├── train/images/ & labels/
│   │   ├── val/images/ & labels/
│   │   ├── test/images/ & labels/
│   │   └── classes.txt
│   ├── coco/
│   │   └── annotations.json
│   ├── unet/
│   │   ├── images/
│   │   └── masks/
│   └── classification/
│       ├── class1/
│       ├── class2/
│       └── ...
├── 📝 scripts/
│   ├── train_yolo.py
│   ├── train_coco.py
│   └── train_unet.py
├── 📋 reports/
│   ├── category_analysis_report.json
│   ├── category_analysis_report.md
│   └── validation_report.json
├── 🎯 models/
└── 📊 results/
```

## 🏷️ Categorías Dentales Soportadas

### Detección y Diagnóstico
- **caries** - Caries, cavidades, decaimiento
- **tooth** - Dientes, molares, premolares, caninos, incisivos
- **filling** - Empastes, restauraciones, amalgama, composite
- **crown** - Coronas, caps, coronas dentales
- **implant** - Implantes, tornillos, fijaciones

### Tratamientos
- **root_canal** - Tratamiento de conducto, endodoncia
- **periapical_lesion** - Lesiones periapicales, abscesos
- **bone_loss** - Pérdida ósea, periodontal
- **impacted** - Dientes impactados, muelas del juicio

### Estructuras Anatómicas
- **maxillary_sinus** - Seno maxilar
- **mandible** - Mandíbula, rama mandibular
- **maxilla** - Maxilar

### Ortodóncicos y Protésicos
- **orthodontic** - Brackets, alambres, aparatos
- **prosthetic** - Dentaduras, prótesis

## 🔧 Configuración

### Clases Unificadas
El sistema mapea automáticamente variaciones de nombres a clases unificadas:

```python
unified_classes = {
    'caries': ['caries', 'Caries', 'CARIES', 'cavity', 'decay', 'Q1_Caries'],
    'tooth': ['tooth', 'teeth', 'Tooth', 'diente', 'molar', 'premolar'],
    # ... más categorías
}
```

### Resoluciones Estándar
```python
standard_resolutions = {
    'yolo': (640, 640),      # Detección YOLO
    'coco': (640, 640),      # Detección COCO  
    'unet': (512, 512),      # Segmentación U-Net
    'classification': (224, 224)  # Clasificación
}
```

## 📊 Análisis de Calidad

### Métricas Calculadas
- **Balance Score** (0-100): Distribución equilibrada de clases
- **Coverage Score** (0-100): Cobertura de categorías dentales
- **Quality Score** (0-100): Calidad general del dataset

### Recomendaciones Automáticas
- **Alta prioridad**: Problemas críticos que requieren atención inmediata
- **Media prioridad**: Mejoras recomendadas
- **Baja prioridad**: Optimizaciones opcionales

## 🔄 Workflow Inteligente

### 1. Escaneo y Análisis
```python
# Escaneo automático de datasets
analysis = manager.scan_and_analyze_datasets()

# Análisis inteligente de categorías  
categories = manager.analyze_categories()
```

### 2. Selección Interactiva
```python
# Mostrar categorías disponibles
manager.show_categories_menu()

# Selección múltiple interactiva
manager.dataset_selection_menu()
```

### 3. Conversión Inteligente
```python
# Conversión a formato específico
manager.convert_to_yolo()
manager.convert_to_coco()
manager.convert_to_unet()

# Conversión múltiple
manager.convert_multiple_formats()
```

### 4. Balanceado Automático
```python
# Análisis de distribución
manager.show_data_distribution()

# Balanceado automático
manager.auto_balance_data()

# Balanceado personalizado
manager.custom_balance_data()
```

### 5. Verificación y Validación
```python
# Verificación completa
validation_result = manager.verify_and_validate()

# Generación de reportes
manager.generate_validation_report()
```

## 📝 Ejemplos de Uso

### Ejemplo Básico
```python
from Src.smart_workflow_manager import SmartDentalWorkflowManager

# Inicializar workflow
manager = SmartDentalWorkflowManager(
    base_path="_dataSets",
    output_path="Dist/dental_ai"
)

# Ejecutar análisis
manager._scan_and_analyze()

# Mostrar categorías
manager._show_categories_menu()

# Workflow completo automático
manager._run_complete_workflow()
```

### Ejemplo Avanzado
```python
# Configuración personalizada
manager = SmartDentalWorkflowManager()

# Personalizar clases unificadas
manager.unified_classes.update({
    'nueva_categoria': ['patron1', 'patron2', 'patron3']
})

# Personalizar resoluciones
manager.standard_resolutions.update({
    'yolo': (1024, 1024)  # Mayor resolución
})

# Ejecutar workflow personalizado
manager.run_interactive_workflow()
```

## 🚨 Troubleshooting

### Errores Comunes

**Error: "No datasets found"**
- Verifica que existe el directorio `_dataSets`
- Asegúrate de que hay subdirectorios con datasets

**Error: "No classes detected"**
- Verifica que los datasets tienen anotaciones
- Revisa el formato de archivos de anotación

**Error: "Conversion failed"**
- Verifica permisos de escritura en `Dist/dental_ai`
- Asegúrate de que hay espacio en disco suficiente

### Dependencias
```bash
pip install -r requirements.txt
```

Dependencias principales:
- `opencv-python` - Procesamiento de imágenes
- `numpy` - Operaciones numéricas
- `pandas` - Análisis de datos
- `matplotlib` - Visualización
- `seaborn` - Gráficos estadísticos
- `tqdm` - Barras de progreso
- `pathlib` - Manejo de rutas

## 🔮 Próximas Funcionalidades

- **Integración con MLflow** para tracking de experimentos
- **API REST** para uso remoto
- **Dashboard web** interactivo
- **Integración con DVC** para versionado de datos
- **Soporte para más formatos** (PASCAL VOC, TensorFlow Records)
- **Análisis de calidad de imágenes** con IA
- **Detección automática de anomalías** en datasets

## 📞 Soporte

Para reportar bugs o solicitar funcionalidades:
- Crear issue en el repositorio
- Incluir logs de error
- Proporcionar información del sistema
- Adjuntar dataset de ejemplo si es posible

---

**Autor**: Anton Sychev  
**Versión**: 3.0 (Smart Interactive)  
**Fecha**: Julio 2025
