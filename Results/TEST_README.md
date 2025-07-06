# 🧪 Sistema de Prueba de Modelos Dentales

Este directorio contiene un sistema completo para probar y gestionar modelos YOLO entrenados para análisis dental.

## 🚀 Scripts Disponibles

### 1. 🏃‍♂️ **Prueba Rápida** - `quick_test.py`
Ejecuta una prueba rápida automática del modelo.

```bash
python quick_test.py
```

**¿Qué hace?**
- 🔍 Busca automáticamente modelos entrenados
- 📸 Encuentra imágenes de prueba
- 🧪 Ejecuta predicciones en 3 imágenes
- 💾 Guarda resultados en `test_results/`

### 2. 🎨 **Demo Visual** - `visual_demo.py`
Crea visualizaciones atractivas de las predicciones.

```bash
python visual_demo.py
```

**Características:**
- 📊 Visualización lado a lado (original vs predicción)
- 🎯 Cajas de detección con colores por clase
- 📈 Gráficos de resumen y estadísticas
- 💾 Resultados en `demo_results/`

### 3. 🔧 **Tester Completo** - `test_model.py`
Sistema completo de pruebas con múltiples opciones.

```bash
# Probar una imagen específica
python test_model.py --model path/to/model.pt --image image.jpg

# Probar conjunto de datos
python test_model.py --model path/to/model.pt --dataset datasets/detection_combined/ --split val

# Benchmark completo
python test_model.py --model path/to/model.pt --dataset datasets/detection_combined/ --benchmark
```

**Opciones:**
- `--model, -m`: Ruta al modelo (.pt)
- `--dataset, -d`: Ruta al dataset
- `--image, -i`: Probar imagen específica
- `--output, -o`: Directorio de salida
- `--benchmark, -b`: Ejecutar benchmark completo
- `--split`: Conjunto a usar (train/val/test)
- `--max-images`: Máximo número de imágenes

### 4. 📦 **Gestor de Modelos** - `model_manager.py`
Organiza y gestiona tus modelos entrenados.

```bash
# Interfaz interactiva
python model_manager.py

# Escanear modelos
python model_manager.py --scan

# Organizar modelo específico
python model_manager.py --organize path/to/model.pt --name mi_modelo
```

## 📁 Estructura de Resultados

```
Dist/dental_ai/
├── test_results/          # Resultados de quick_test
│   ├── predicted_*.jpg    # Imágenes con predicciones
│   └── test_summary.json  # Resumen de resultados
├── demo_results/          # Resultados de visual_demo
│   ├── demo_01_*.png      # Visualizaciones
│   └── detection_summary.png
├── models/                # Modelos organizados
│   └── yolo_detect/
│       ├── modelo_1/
│       └── modelo_2/
└── *.py                  # Scripts de prueba
```

## 🎯 Guía de Uso Paso a Paso

### **Paso 1: Verificar que tienes un modelo**
```bash
# Buscar modelos automáticamente
python model_manager.py --scan
```

### **Paso 2: Prueba rápida**
```bash
# Ejecutar prueba automática
python quick_test.py
```

### **Paso 3: Demo visual (opcional)**
```bash
# Crear visualizaciones bonitas
python visual_demo.py
```

### **Paso 4: Análisis detallado**
```bash
# Benchmark completo
python test_model.py --model path/to/best.pt --dataset datasets/detection_combined/ --benchmark
```

## 📊 Interpretación de Resultados

### **Métricas YOLO**
- **mAP50**: Precisión promedio a IoU=0.5 (>0.5 = bueno, >0.7 = excelente)
- **mAP50-95**: Precisión promedio en rango IoU 0.5-0.95 (más estricto)
- **Precisión**: Qué porcentaje de detecciones son correctas
- **Recall**: Qué porcentaje de objetos reales fueron detectados

### **Confianza**
- **>0.8**: Alta confianza (muy buenas detecciones)
- **0.5-0.8**: Confianza media (buenas detecciones)
- **<0.5**: Baja confianza (revisar detecciones)

## 🔧 Instalación de Dependencias

```bash
pip install ultralytics matplotlib seaborn opencv-python PyYAML
```

## 🚨 Troubleshooting

### **"No se encontraron modelos"**
1. Verifica que entrenaste un modelo
2. Ejecuta el entrenamiento:
   ```bash
   cd datasets/detection_combined
   ./train_dental_dataset.sh
   ```

### **"ultralytics no está instalado"**
```bash
pip install ultralytics
```

### **"No se encontraron imágenes de prueba"**
1. Verifica que el dataset esté completo
2. Ejecuta el workflow principal para generar datos

### **Error de memoria**
- Reduce el número de imágenes: `--max-images 5`
- Usa imagen más pequeña en el modelo

## 🎨 Personalización

### **Cambiar colores de visualización**
Edita `visual_demo.py`:
```python
self.colors = sns.color_palette("your_palette", 20)
```

### **Añadir nuevas métricas**
Edita `test_model.py` y añade en el método `benchmark_model()`.

### **Cambiar umbrales de confianza**
```python
# En test_model.py
results = self.model(str(image_path), conf=0.5)  # Cambiar 0.5
```

## 📞 Soporte

Si tienes problemas:
1. Verifica las dependencias
2. Revisa que el modelo existe
3. Confirma que las imágenes de prueba están disponibles
4. Ejecuta `python model_manager.py --scan` para diagnóstico

---

## 🎉 ¡Disfruta probando tu modelo!

Estos scripts te permitirán:
- ✅ Validar que tu modelo funciona correctamente
- 📊 Obtener métricas de rendimiento
- 🎨 Crear visualizaciones profesionales
- 📦 Organizar tus modelos entrenados

¡Tu modelo dental está listo para detectar caries, dientes, empastes y más! 🦷
