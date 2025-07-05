# 🦷 Dental AI Workflow Manager v2.0

Sistema modular para análisis, procesamiento y preparación de datasets dentales para entrenamiento de modelos de IA.

## 🚀 Inicio Rápido

### Análisis Rápido de Datasets
```bash
# Análisis completo con visualizaciones
python demo_herramientas.py --analisis

# Ver resumen de resultados
python demo_herramientas.py --resumen
```

### Sistema Modular v2.0
```bash
# Workflow completo modular
python ejemplo_uso_v2.py

# Análisis rápido
python ejemplo_uso_v2.py --quick
```

### Sistema Legacy (compatible)
```bash
# Menú interactivo legacy
python DataWorkflowManager.py

# Ejemplos legacy
python ejemplo_uso.py completo
python ejemplo_uso.py v2  # Ver nuevas herramientas
```

## 📁 Estructura del Proyecto

```
Dental-AI-Workflow/
├── 📊 StatisticsResults/     # Resultados de análisis y estadísticas
│   ├── *.png                # Gráficos y visualizaciones
│   ├── *.csv                # Tablas de datos
│   ├── *.json               # Datos de análisis
│   └── *.html               # Dashboards interactivos
│
├── 🏗️ Src/                   # Módulos principales (v2.0)
│   ├── workflow_manager.py  # Gestor principal del workflow
│   ├── data_analyzer.py     # Analizador de datos
│   ├── data_processor.py    # Procesador de datasets
│   ├── structure_generator.py # Generador de estructuras
│   └── script_templates.py  # Plantillas de scripts
│
├── 🔧 Utils/                 # Herramientas y utilidades
│   ├── advanced_analysis.py # 🔥 Análisis avanzado integrado
│   ├── visualization.py     # Visualización de datos
│   ├── data_augmentation.py # Augmentación de datos
│   └── dental_format_converter.py # Convertidor de formatos
│
├── 📚 Wiki/                  # Documentación centralizada
│   ├── README.md            # Índice de documentación
│   ├── USAGE_EXAMPLES.md    # Ejemplos de uso
│   ├── API_REFERENCE.md     # Referencia de la API
│   ├── WORKFLOW_GUIDE.md    # Guía del workflow
│   └── MIGRACION_V2.md      # Guía de migración
│
├── 🎯 Dist/dental_ai/        # Estructura de salida final
│   ├── datasets/            # Datasets procesados
│   ├── models/              # Modelos entrenados
│   ├── scripts/             # Scripts de entrenamiento
│   └── docs/                # Documentación generada
│
├── main.py                  # Punto de entrada principal
├── ejemplo_uso_v2.py        # 🔥 Ejemplo modular v2.0
├── demo_herramientas.py     # 🔥 Demo de herramientas integradas
├── requirements.txt         # Dependencias del proyecto
└── ejemplo_uso.py           # Ejemplo legacy (compatible)
```

## 🔥 Nuevas Características v2.0

### ✅ Herramientas Integradas
- **Archivos legacy** → Integrados en `Utils/advanced_analysis.py`
- Resultados centralizados en `StatisticsResults/`
- Documentación centralizada en `Wiki/`

### ✅ Análisis Avanzado
- Análisis de calidad de imágenes
- Detección automática de formatos
- Visualizaciones interactivas
- Dashboards HTML
- Reportes en múltiples formatos

### ✅ Sistema Modular
- Código separado en módulos
- Estructura clara y mantenible
- API bien documentada
- Herramientas reutilizables

## 📊 Análisis y Estadísticas

### Ejecutar Análisis Completo
```python
from Utils.advanced_analysis import analyze_dental_datasets

# Análisis completo con visualizaciones
results = analyze_dental_datasets("_dataSets", "StatisticsResults")
```

### Archivos Generados
```
StatisticsResults/
├── dental_dataset_analysis.json    # Datos completos
├── dataset_overview.png            # Resumen visual
├── format_distribution.png         # Distribución por formato
├── categories_analysis.png         # Análisis de categorías
├── quality_analysis.png            # Análisis de calidad
├── size_distribution.png           # Distribución de tamaños
├── datasets_summary_table.csv      # Tabla resumen
├── dataset_report.md               # Reporte en texto
└── dental_datasets_dashboard.html  # 🌐 Dashboard interactivo
```

## 🎯 Casos de Uso

### 🔬 Investigación Académica
```bash
# Análisis comparativo de datasets
python demo_herramientas.py --analisis

# Generar reportes para publicaciones
python demo_herramientas.py --visuales
```

### 🏭 Desarrollo Comercial
```bash
# Preparar datasets para entrenamiento
python ejemplo_uso_v2.py

# Validar calidad de datos
python demo_herramientas.py --resumen
```

### 🎓 Educación
```bash
# Explorar datasets dentales
python ejemplo_uso.py v2

# Aprender sobre análisis de datos
python demo_herramientas.py --analisis
```

## 🛠️ Instalación y Configuración

### Dependencias
```bash
# Instalar dependencias completas
pip install -r requirements.txt

# O dependencias básicas
pip install -r requirements.txt
```

### Configuración Inicial
```bash
# Verificar estructura
python ejemplo_uso.py verificar

# Ver herramientas disponibles
python ejemplo_uso.py v2

# Ejecutar demo
python demo_herramientas.py
```

## 📚 Documentación

### Guides y Tutoriales
- [`Wiki/README.md`](Wiki/README.md) - Índice completo de documentación
- [`Wiki/USAGE_EXAMPLES.md`](Wiki/USAGE_EXAMPLES.md) - Ejemplos detallados
- [`Wiki/WORKFLOW_GUIDE.md`](Wiki/WORKFLOW_GUIDE.md) - Guía del workflow

### Referencias Técnicas
- [`Wiki/API_REFERENCE.md`](Wiki/API_REFERENCE.md) - Referencia completa de la API
- [`Wiki/MIGRACION_V2.md`](Wiki/MIGRACION_V2.md) - Guía de migración

## 🔄 Migración desde v1.0

### Usuarios Existentes
1. Las herramientas legacy siguen funcionando
2. Los nuevos módulos ofrecen más funcionalidades
3. Resultados ahora centralizados en `StatisticsResults/`
4. Documentación actualizada en `Wiki/`

### Comandos de Migración
```bash
# Ver estado actual
python ejemplo_uso.py verificar

# Conocer nuevas herramientas
python ejemplo_uso.py v2

# Probar sistema v2.0
python ejemplo_uso_v2.py --quick
```

## 🤝 Contribución

### Reportar Problemas
- Crear issue con logs detallados
- Incluir configuración del sistema
- Describir pasos para reproducir

### Desarrollo
- Fork del repositorio
- Crear rama para features
- Seguir convenciones de código
- Enviar pull request

## 📞 Soporte

- **Documentación**: [`Wiki/`](Wiki/)
- **Ejemplos**: [`Wiki/USAGE_EXAMPLES.md`](Wiki/USAGE_EXAMPLES.md)
- **API**: [`Wiki/API_REFERENCE.md`](Wiki/API_REFERENCE.md)
- **Demo**: `python demo_herramientas.py`

## 📈 Estado del Proyecto

### ✅ Completado
- [x] Sistema modular v2.0
- [x] Integración de herramientas de análisis
- [x] Centralización de resultados
- [x] Documentación completa
- [x] Ejemplos de uso
- [x] Compatibilidad con v1.0

### 🔄 En Desarrollo
- [ ] Interfaz web interactiva
- [ ] API REST
- [ ] Integración con MLflow
- [ ] Soporte para más formatos

---

**Versión**: 2.0 (Modular)  
**Autor**: Anton Sychev  
**Licencia**: MIT  
**Última actualización**: 15 de enero de 2025
