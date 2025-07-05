# 🧠 Guía Rápida: Smart Dental AI Workflow Manager v3.0

## 🎯 ¿Qué es este sistema?

Es un **sistema inteligente** que te ayuda a:
- 📊 Analizar automáticamente tus datasets dentales
- 🔄 Convertir entre diferentes formatos (YOLO, COCO, U-Net)
- ⚖️ Balancear y preparar datos para entrenar modelos de IA
- 📝 Generar scripts de entrenamiento listos para usar

## 🚀 Cómo usarlo

### **Opción 1: Modo Interactivo (Recomendado)**
```bash
python smart_dental_workflow.py
```
Te guía paso a paso con un menú interactivo.

### **Opción 2: Análisis Rápido**
```bash
python smart_dental_workflow.py --mode analysis
```
Solo escanea y analiza tus datasets sin procesarlos.

### **Opción 3: Workflow Automático**
```bash
python smart_dental_workflow.py --mode auto
```
Ejecuta todo el proceso automáticamente.

## 📁 Tu estructura actual

Detectamos **41 datasets** organizados así:
```
_dataSets/
├── _YOLO/ (15 datasets)
├── _COCO/ (10 datasets) 
├── _pure images and masks/ (14 datasets)
└── _UNET/ (2 datasets)
```

## 🎮 Menú Interactivo

Cuando ejecutes el modo interactivo verás opciones como:

1. **🔍 Escanear y analizar datasets**
   - Detecta automáticamente todas las categorías
   - Analiza calidad de imágenes
   - Cuenta muestras por clase

2. **📊 Ver categorías disponibles**
   - Muestra todas las clases detectadas
   - Estadísticas por categoría
   - Recomendaciones de balanceo

3. **📦 Seleccionar datasets**
   - Elige qué datasets incluir
   - Combina múltiples fuentes
   - Control granular de selección

4. **🔄 Convertir formatos**
   - YOLO para detección de objetos
   - COCO para segmentación
   - U-Net para segmentación médica
   - Clasificación por carpetas

5. **⚖️ Balancear datos**
   - Detección automática de desbalance
   - Augmentación inteligente
   - División train/val/test

6. **✅ Verificar y validar**
   - Control de calidad
   - Verificación de integridad
   - Reportes de validación

7. **📝 Generar scripts**
   - Scripts de entrenamiento personalizados
   - Configuraciones optimizadas
   - Ready-to-run código

## 📊 Categorías Dentales Soportadas

El sistema reconoce automáticamente:
- **Caries**: caries, decay, cavity, hole
- **Implantes**: implant, screw, titanium, crown
- **Ortodónticos**: bracket, wire, brace, retainer
- **Periodontales**: gingivitis, periodontitis
- **Endodónticos**: root_canal, pulp
- **Prótesis**: denture, bridge, partial
- **Anomalías**: impacted, supernumerary
- **Y muchas más...**

## 📁 Resultados

Los datos procesados se guardan en:
```
Dist/dental_ai/
├── datasets/          # Datasets convertidos y listos
├── scripts/           # Scripts de entrenamiento
├── reports/           # Análisis y métricas
└── analysis/          # Visualizaciones y estadísticas
```

## 💡 Recomendación de uso

**Para empezar:**
1. Ejecuta `python smart_dental_workflow.py`
2. Selecciona "🔍 Escanear y analizar datasets"
3. Revisa las categorías detectadas
4. Selecciona los datasets que quieres usar
5. Elige el formato de salida (YOLO recomendado para empezar)
6. Deja que el sistema balance automáticamente
7. Genera los scripts de entrenamiento

## 🔧 Uso Programático

También puedes usar el sistema desde código:

```python
from Src.smart_workflow_manager import SmartDentalWorkflowManager

# Inicializar
manager = SmartDentalWorkflowManager(
    base_path="_dataSets",
    output_path="Dist/dental_ai"
)

# Análisis automático
manager._scan_and_analyze()

# Ver categorías
manager._show_categories_menu()

# Workflow completo
manager._run_complete_workflow()
```

## 🚨 Troubleshooting

**Si ves errores:**
1. Verifica dependencias: `pip install -r requirements.txt`
2. Verifica permisos en la carpeta `Dist/`
3. Asegúrate de que `_dataSets/` existe y tiene contenido

**Para más ayuda:**
- 📚 `README_SMART.md` - Documentación completa
- 📖 `SMART_WORKFLOW_GUIDE.md` - Guía detallada
- 🎪 `python demo_smart_workflow.py` - Demo interactiva

---

**¡Tienes 41 datasets listos para procesar! 🎉**
**Ejecuta: `python smart_dental_workflow.py` para empezar**
