# 🦷 Dental AI Workflow Manager v3.0

> Sistema inteligente para gestión, análisis y preparación de datasets dentales con IA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 🚀 Novedades v3.0

### 🧠 **Smart Workflow Manager**
- **Análisis automático** de categorías dentales
- **Detección inteligente** de patrones en datos
- **Menú interactivo** para gestión completa
- **Mapeo unificado** de clases dentales
- **Verificación de calidad** automática

### 🎯 **Gestión Avanzada de Categorías**
- **Detección automática** de 12+ categorías dentales
- **Mapeo inteligente** con patrones regex
- **Análisis de distribución** y balance
- **Recomendaciones automáticas** de mejora
- **Soporte extensible** para nuevas categorías

### 🔄 **Conversión Múltiple**
- **YOLO** para detección de objetos
- **COCO** para detección y segmentación  
- **U-Net** para segmentación médica
- **Clasificación** por directorios
- **Conversión batch** automática

## 📋 Tabla de Contenidos

- [🎯 Características](#-características)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [📁 Estructura](#-estructura)
- [🎮 Modos de Uso](#-modos-de-uso)
- [🏷️ Categorías Soportadas](#️-categorías-soportadas)
- [📊 Análisis de Calidad](#-análisis-de-calidad)
- [🔧 Configuración](#-configuración)
- [📝 Ejemplos](#-ejemplos)
- [🚨 Troubleshooting](#-troubleshooting)

## 🎯 Características

### 🔍 **Análisis Inteligente**
- ✅ Escaneo automático de datasets
- ✅ Detección de formatos (YOLO, COCO, U-Net, Clasificación)
- ✅ Análisis de calidad con métricas automáticas
- ✅ Detección de patrones en nombres de clases
- ✅ Mapeo inteligente a categorías unificadas

### 📊 **Gestión de Datos**
- ✅ Selección interactiva de datasets
- ✅ Balanceado automático e inteligente
- ✅ Augmentación dirigida por categoría
- ✅ Verificación de integridad de datos
- ✅ Distribución automática train/val/test

### 🔄 **Conversión y Preparación**
- ✅ Conversión a múltiples formatos simultánea
- ✅ Normalización de resoluciones
- ✅ Generación de metadatos automática
- ✅ Scripts de entrenamiento listos para usar
- ✅ Estructura optimizada para producción

### 📋 **Reportes y Validación**
- ✅ Reportes detallados en JSON y Markdown
- ✅ Métricas de calidad automáticas
- ✅ Recomendaciones de mejora
- ✅ Validación de resultados
- ✅ Tracking de conversiones

## 🚀 Inicio Rápido

### 1. **Instalación**
```bash
# Clonar repositorio
git clone <repository-url>
cd XRAY

# Instalar dependencias
pip install -r requirements.txt
```

### 2. **Estructura de Datos**
Organiza tus datasets en la estructura:
```
_dataSets/
├── _YOLO/
│   ├── dataset1/
│   └── dataset2/
├── _COCO/
│   ├── dataset3/
│   └── dataset4/
├── _pure images and masks/
│   ├── dataset5/
│   └── dataset6/
└── _UNET/
    ├── dataset7/
    └── dataset8/
```

### 3. **Ejecución Rápida**

#### **🎮 Modo Interactivo (Recomendado)**
```bash
python smart_dental_workflow.py
```

#### **🤖 Modo Automático**
```bash
python smart_dental_workflow.py --mode auto
```

#### **⚡ Análisis Rápido**
```bash
python smart_dental_workflow.py --mode analysis
```

#### **🎪 Demo Completa**
```bash
python demo_smart_workflow.py
```

### 4. **Resultados**
Los resultados se generan en:
```
Dist/dental_ai/
├── datasets/          # Datasets convertidos
├── scripts/           # Scripts de entrenamiento
├── reports/           # Reportes de análisis
└── analysis/          # Datos de análisis
```

## 📁 Estructura del Proyecto

```
XRAY/
├── 🧠 smart_dental_workflow.py     # Sistema principal v3.0
├── 🎪 demo_smart_workflow.py       # Demo interactiva
├── 📝 ejemplo_uso_v2.py            # Ejemplos extendidos
├── 📚 SMART_WORKFLOW_GUIDE.md      # Guía completa v3.0
├── 📋 README.md                    # Este archivo
├── 📦 requirements.txt             # Dependencias
├── 
├── 🗂️ Src/                         # Módulos principales
│   ├── 🧠 smart_workflow_manager.py      # Manager v3.0
│   ├── 🧩 smart_category_analyzer.py     # Analizador inteligente
│   ├── 📊 data_analyzer.py               # Análisis de datos
│   ├── 🔄 data_processor.py              # Procesamiento
│   ├── 🏗️ structure_generator.py         # Generador de estructura
│   └── 📝 script_templates.py            # Templates de scripts
├── 
├── 🛠️ Utils/                       # Utilidades
│   ├── 📊 visualization.py               # Visualización
│   ├── ⚖️ data_augmentation.py           # Augmentación
│   └── 🔄 dental_format_converter.py     # Conversores
├── 
├── 📊 Dist/                        # Resultados (auto-generado)
│   └── dental_ai/                        # Salida principal
├── 
└── 📁 _dataSets/                   # Datasets originales
    ├── _YOLO/
    ├── _COCO/
    ├── _pure images and masks/
    └── _UNET/
```

## 🎮 Modos de Uso

### **1. 🎮 Modo Interactivo Completo**

El modo más completo con menú paso a paso:

```bash
python smart_dental_workflow.py
```

**Menú principal:**
- 🔍 **Escanear y analizar** datasets
- 📊 **Ver categorías** disponibles  
- 📦 **Seleccionar datasets** interactivamente
- 🔄 **Convertir formatos** (YOLO/COCO/U-Net)
- ⚖️ **Balancear datasets** automática o manualmente
- ✅ **Verificar y validar** resultados
- 📝 **Generar scripts** de entrenamiento
- 🚀 **Workflow completo** automático
- 📋 **Reporte de análisis** detallado

### **2. 🤖 Modo Automático**

Ejecuta todo el pipeline sin intervención:

```bash
python smart_dental_workflow.py --mode auto
```

**Proceso automático:**
1. Escaneo y análisis de datasets
2. Selección automática (categorías con >10 muestras)
3. Conversión a múltiples formatos
4. Balanceado automático
5. Verificación completa
6. Generación de scripts

### **3. ⚡ Modo Análisis Rápido**

Solo análisis sin conversión:

```bash
python smart_dental_workflow.py --mode analysis
```

**Incluye:**
- Escaneo rápido de datasets
- Detección de categorías
- Reporte de resumen
- Métricas básicas de calidad

### **4. 📝 Ejemplos Extendidos**

Sistema de ejemplos paso a paso:

```bash
python ejemplo_uso_v2.py
```

**Opciones disponibles:**
1. 📝 Ejemplo básico
2. 🔄 Procesamiento avanzado
3. ⚙️ Configuración personalizada
4. 🧩 Uso modular de componentes
5. 🧠 Smart Workflow (NUEVO)
6. 🚀 Workflow automático completo (NUEVO)

### **5. 🎪 Demo Interactiva**

Demostración completa con explicaciones:

```bash
python demo_smart_workflow.py
```

**Incluye:**
- Demo paso a paso con explicaciones
- Simulación de proceso completo
- Explicación de cada fase
- Recomendaciones y próximos pasos

## 🏷️ Categorías Dentales Soportadas

### **🔍 Detección y Diagnóstico**
| Categoría | Variantes Detectadas | Descripción |
|-----------|---------------------|-------------|
| **caries** | caries, cavity, decay, Q1_Caries, etc. | Caries dentales y cavidades |
| **tooth** | tooth, teeth, molar, premolar, canine, incisor | Dientes y estructuras dentales |
| **filling** | filling, restoration, amalgam, composite | Empastes y restauraciones |
| **crown** | crown, cap, corona | Coronas dentales |
| **implant** | implant, screw, fixture | Implantes dentales |

### **🏥 Tratamientos**
| Categoría | Variantes Detectadas | Descripción |
|-----------|---------------------|-------------|
| **root_canal** | root canal, endodontic, treated tooth | Tratamientos de conducto |
| **periapical_lesion** | periapical lesion, abscess, Q1_Periapical | Lesiones periapicales |
| **bone_loss** | bone loss, alveolar, periodontal | Pérdida ósea |
| **impacted** | impacted, wisdom, third molar | Dientes impactados |

### **🦴 Estructuras Anatómicas**
| Categoría | Variantes Detectadas | Descripción |
|-----------|---------------------|-------------|
| **maxillary_sinus** | maxillary sinus, seno maxilar | Seno maxilar |
| **mandible** | mandible, rama, inferior border | Mandíbula |
| **maxilla** | maxilla, maxilar | Maxilar |

### **🔧 Ortodóncicos y Protésicos**
| Categoría | Variantes Detectadas | Descripción |
|-----------|---------------------|-------------|
| **orthodontic** | bracket, brace, wire, orthodontic | Aparatos ortodóncicos |
| **prosthetic** | denture, prosthetic, artificial | Prótesis dentales |

## 📊 Análisis de Calidad

### **📈 Métricas Automáticas**

| Métrica | Rango | Descripción |
|---------|-------|-------------|
| **Balance Score** | 0-100 | Equilibrio de distribución entre clases |
| **Coverage Score** | 0-100 | Cobertura de categorías dentales |
| **Quality Score** | 0-100 | Calidad general del dataset |
| **Annotation Ratio** | 0-1 | Ratio de imágenes con anotaciones |

### **🎯 Criterios de Calidad**

**📊 Balance Score:**
- **90-100**: Excelente distribución
- **75-89**: Buena distribución  
- **50-74**: Distribución moderada
- **< 50**: Requiere balanceado

**🏷️ Coverage Score:**
- **80-100**: Cobertura completa
- **60-79**: Buena cobertura
- **40-59**: Cobertura moderada
- **< 40**: Cobertura limitada

### **💡 Recomendaciones Automáticas**

**🔴 Alta Prioridad:**
- Dataset muy desbalanceado
- Dataset muy pequeño (<100 muestras)
- Baja cobertura de categorías

**🟡 Media Prioridad:**
- Dataset moderadamente desbalanceado
- Dataset pequeño (<500 muestras)
- Clases no mapeadas

**🟢 Baja Prioridad:**
- Optimizaciones menores
- Mejoras de estructura
- Recomendaciones de formato

## 🔧 Configuración

### **🏷️ Clases Unificadas**

Personaliza el mapeo de clases:

```python
# Extender clases existentes
manager.unified_classes.update({
    'nueva_categoria': ['patron1', 'patron2', 'patron3'],
    'otra_categoria': ['variante_a', 'variante_b']
})

# Agregar patrones regex personalizados
manager.dental_patterns.update({
    'categoria_custom': [r'patron.*regex', r'otro.*patron']
})
```

### **📏 Resoluciones Estándar**

Configurar resoluciones por formato:

```python
manager.standard_resolutions.update({
    'yolo': (1024, 1024),      # Mayor resolución YOLO
    'coco': (1280, 1280),      # Alta resolución COCO
    'unet': (768, 768),        # Resolución U-Net personalizada
    'classification': (299, 299)  # ImageNet estándar
})
```

### **⚙️ Configuración de Workflow**

Ajustar parámetros del proceso:

```python
manager.workflow_config.update({
    'train_ratio': 0.8,         # 80% entrenamiento
    'val_ratio': 0.15,          # 15% validación  
    'test_ratio': 0.05,         # 5% prueba
    'min_samples_per_class': 20, # Mínimo por clase
    'max_augmentation_factor': 5 # Factor de augmentación
})
```

## 📝 Ejemplos

### **🚀 Ejemplo Básico**

```python
from Src.smart_workflow_manager import SmartDentalWorkflowManager

# Inicializar
manager = SmartDentalWorkflowManager(
    base_path="_dataSets",
    output_path="Dist/dental_ai"
)

# Ejecutar workflow completo
manager.run_interactive_workflow()
```

### **🔧 Ejemplo Personalizado**

```python
# Configuración avanzada
manager = SmartDentalWorkflowManager()

# Personalizar clases
manager.unified_classes.update({
    'ortodontico': ['bracket', 'brace', 'alambre'],
    'protesis': ['denture', 'artificial_tooth']
})

# Configurar resoluciones altas
manager.standard_resolutions.update({
    'yolo': (1280, 1280),
    'unet': (1024, 1024)
})

# Ejecutar análisis
manager._scan_and_analyze()
manager._show_categories_menu()
```

### **🤖 Ejemplo Programático**

```python
# Workflow automático programático
manager = SmartDentalWorkflowManager()

# 1. Análisis
analysis = manager._scan_and_analyze()

# 2. Selección automática (categorías con >50 muestras)
manager.selected_datasets = {
    cat: info for cat, info in manager.available_categories.items()
    if info['total_samples'] >= 50
}

# 3. Conversión específica
manager._convert_to_yolo()

# 4. Balanceado automático
manager._auto_balance_data()

# 5. Validación
manager._verify_and_validate()
```

## 🚨 Troubleshooting

### **❌ Errores Comunes**

**"No datasets found"**
```bash
# Verificar estructura
ls -la _dataSets/
ls -la _dataSets/_YOLO/
ls -la _dataSets/_COCO/
```

**"No classes detected"**
```bash
# Verificar anotaciones
find _dataSets/ -name "*.txt" | head -5
find _dataSets/ -name "*.json" | head -5
```

**"Permission denied"**
```bash
# Verificar permisos
chmod +x smart_dental_workflow.py
mkdir -p Dist/dental_ai
```

**"Module not found"**
```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### **🔧 Diagnóstico**

**Verificar instalación:**
```python
# Test básico
python -c "from Src.smart_workflow_manager import SmartDentalWorkflowManager; print('✅ OK')"
```

**Verificar datos:**
```bash
# Ejecutar análisis rápido
python smart_dental_workflow.py --mode analysis
```

**Logs de debug:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ejecutar con logs detallados
manager = SmartDentalWorkflowManager()
manager._scan_and_analyze()
```

### **💡 Optimización de Rendimiento**

**Para datasets grandes:**
```python
# Reducir muestra de análisis
manager.analyzer.sample_size = 100  # Solo 100 archivos por dataset

# Ejecutar en paralelo
manager.parallel_processing = True
manager.max_workers = 4
```

**Para memoria limitada:**
```python
# Procesar por lotes
manager.batch_size = 50
manager.enable_memory_optimization = True
```

## 📚 Documentación Adicional

- 📖 **[SMART_WORKFLOW_GUIDE.md](SMART_WORKFLOW_GUIDE.md)** - Guía completa v3.0
- 📋 **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Guía original
- 🔧 **[API_REFERENCE.md](Wiki/API_REFERENCE.md)** - Referencia de API
- 🎯 **[DENTAL_AI_GUIDE.md](Wiki/DENTAL_AI_GUIDE.md)** - Guía de IA dental
- 📝 **[USAGE_EXAMPLES.md](Wiki/USAGE_EXAMPLES.md)** - Ejemplos de uso

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Anton Sychev** - *Desarrollo inicial y v3.0* - [@antonsychev](https://github.com/antonsychev)

## 🙏 Agradecimientos

- Comunidad de IA médica
- Datasets públicos de radiografías dentales
- Contribuidores del proyecto

---

**📞 Soporte:** [Crear Issue](../../issues/new)  
**📧 Contacto:** [Email](mailto:contact@example.com)  
**🌟 Star:** Si te gusta el proyecto, ¡dale una estrella!

[![GitHub stars](https://img.shields.io/github/stars/usuario/repo.svg?style=social)](https://github.com/usuario/repo/stargazers)
