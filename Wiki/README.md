# 📚 Wiki - Dental AI Workflow Manager

## 📋 Índice de Documentación

### 🔧 Guías de Configuración
- [`WORKFLOW_GUIDE.md`](WORKFLOW_GUIDE.md) - Guía completa del workflow manager
- [`MIGRACION_V2.md`](MIGRACION_V2.md) - Guía de migración a la versión 2.0
- [`IMPLEMENTACION_COMPLETADA.md`](IMPLEMENTACION_COMPLETADA.md) - Estado de implementación

### 📊 Análisis y Estadísticas
- [`DENTAL_AI_GUIDE.md`](DENTAL_AI_GUIDE.md) - Guía del sistema de análisis dental
- [`dental_dataset_report.md`](dental_dataset_report.md) - Reporte de análisis de datasets

### 🎯 Guías de Uso
- [`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md) - Ejemplos de uso del sistema
- [`API_REFERENCE.md`](API_REFERENCE.md) - Referencia de la API

### 🚀 Inicio Rápido

#### Para Usuarios Nuevos
1. Leer [`DENTAL_AI_GUIDE.md`](DENTAL_AI_GUIDE.md) para entender el sistema
2. Seguir [`WORKFLOW_GUIDE.md`](WORKFLOW_GUIDE.md) para configurar el workflow
3. Revisar [`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md) para ejemplos prácticos

#### Para Migración desde v1.0
1. Revisar [`MIGRACION_V2.md`](MIGRACION_V2.md) 
2. Ejecutar script de migración
3. Validar nueva estructura

## 📁 Estructura del Proyecto

```
Dental-AI-Workflow/
├── Src/                     # Módulos principales
│   ├── workflow_manager.py  # Gestor principal del workflow
│   ├── data_analyzer.py     # Analizador de datos
│   ├── data_processor.py    # Procesador de datasets
│   └── structure_generator.py # Generador de estructuras
├── Utils/                   # Utilidades y herramientas
│   ├── advanced_analysis.py # Análisis avanzado integrado
│   ├── visualization.py     # Visualización de datos
│   ├── data_augmentation.py # Augmentación de datos
│   └── dental_format_converter.py # Convertidor de formatos
├── StatisticsResults/       # Resultados de análisis
│   ├── *.png               # Gráficos y visualizaciones
│   ├── *.csv               # Tablas de datos
│   └── *.html              # Dashboards interactivos
├── Wiki/                    # Documentación centralizada
├── Dist/dental_ai/         # Estructura de salida final
└── main.py                 # Punto de entrada principal
```

## 🔄 Flujo de Trabajo Típico

### 1. Análisis de Datasets
```bash
# Ejecutar análisis completo
python -c "from Utils.advanced_analysis import analyze_dental_datasets; analyze_dental_datasets('_dataSets')"

# O usando el main
python main.py --analyze
```

### 2. Creación de Estructura
```bash
# Workflow completo
python main.py --full-workflow

# Solo estructura base
python main.py --create-structure
```

### 3. Procesamiento de Datasets
```bash
# Procesar datasets específicos
python main.py --process-datasets --format yolo

# Generar scripts de entrenamiento
python main.py --generate-scripts
```

## 📊 Análisis y Estadísticas

El sistema genera automáticamente:

### Visualizaciones
- **Distribución por formato**: Gráficos de datasets por tipo
- **Análisis de categorías**: WordCloud y frecuencias
- **Calidad de imágenes**: Distribución de calidad
- **Tamaños de imágenes**: Análisis de resoluciones

### Reportes
- **CSV detallado**: Tabla completa de todos los datasets
- **Dashboard HTML**: Visualización interactiva
- **Reporte en texto**: Resumen ejecutivo

### Archivos Generados
```
StatisticsResults/
├── dental_dataset_analysis.json    # Datos completos del análisis
├── dataset_overview.png            # Resumen visual
├── format_distribution.png         # Distribución por formato
├── categories_analysis.png         # Análisis de categorías
├── quality_analysis.png            # Análisis de calidad
├── size_distribution.png           # Distribución de tamaños
├── datasets_summary_table.csv      # Tabla resumen
├── dataset_report.md               # Reporte en texto
└── dental_datasets_dashboard.html  # Dashboard interactivo
```

## 🛠️ Herramientas Disponibles

### Módulos Principales
- **workflow_manager.py**: Gestión completa del workflow
- **data_analyzer.py**: Análisis profundo de datasets
- **data_processor.py**: Procesamiento y conversión
- **structure_generator.py**: Creación de estructuras

### Utilidades Avanzadas
- **advanced_analysis.py**: Análisis estadístico completo
- **visualization.py**: Generación de gráficos
- **data_augmentation.py**: Balanceo y augmentación
- **dental_format_converter.py**: Conversión entre formatos

## 🎯 Casos de Uso

### Investigación Académica
- Análisis comparativo de datasets
- Evaluación de calidad de datos
- Preparación para publicaciones

### Desarrollo Comercial
- Preparación de datasets para entrenamiento
- Validación de calidad de datos
- Optimización de pipelines

### Educación
- Exploración de datasets dentales
- Aprendizaje de técnicas de análisis
- Demostración de workflows

## 📞 Soporte y Contribución

### Reportar Problemas
- Crear issue en el repositorio
- Incluir logs y archivos de configuración
- Describir pasos para reproducir

### Contribuir
- Fork del repositorio
- Crear rama para feature
- Enviar pull request

### Contacto
- **Autor**: Anton Sychev
- **Email**: anton@sychev.xyz
- **Versión**: 2.0 (Modular)

---

*Última actualización: 15 de enero de 2025*
