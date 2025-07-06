# 🚀 Testing de Modelos Sin CUDA/PyTorch

**Para máquinas locales sin GPU que necesitan probar modelos entrenados en máquinas con GPU**

## 🎯 Caso de Uso

- ✅ **Máquina de entrenamiento**: Tiene CUDA, GPU, PyTorch, Ultralytics
- 🖥️ **Máquina local**: Sin CUDA, sin PyTorch (tu caso)
- 🔄 **Objetivo**: Probar modelos entrenados usando transferencia de archivos

## 🛠️ Herramientas Disponibles

### 1. `workflow_no_torch.py` - Gestor Principal
```bash
python workflow_no_torch.py
```
**Funciones:**
- 📋 Lista modelos disponibles
- 📊 Lista datasets disponibles  
- 📦 Crea lotes de prueba para GPU
- 📊 Procesa resultados de GPU

### 2. `model_tools_no_torch.py` - Herramientas Específicas
```bash
# Inspeccionar modelo (sin cargar PyTorch)
python model_tools_no_torch.py --action inspect --model models/tu_modelo.pt

# Preparar lote de prueba
python model_tools_no_torch.py --action prepare --dataset datasets/detection_combined --output batches/test_batch

# Procesar resultados de GPU
python model_tools_no_torch.py --action process --results results/gpu_output/

# Generar script para GPU
python model_tools_no_torch.py --action generate --output scripts/
```

## 🔄 Flujo de Trabajo Recomendado

### Paso 1: Preparar Lote Local 🖥️
```bash
cd /Volumes/3TB/Ai/XRAY/Dist/dental_ai
python workflow_no_torch.py
# Selecciona opción 3: Crear lote de prueba
```

Esto creará:
```
batches/modelo_dataset_20250706_123456/
├── batch_info.json      # Metadata del lote
├── data.yaml           # Configuración del dataset
├── README.md           # Instrucciones
└── val/                # Imágenes de prueba
    ├── val_001_*.jpg
    ├── val_002_*.jpg
    └── ...
```

### Paso 2: Transferir a Máquina GPU 🚀
```bash
# Comprimir lote (opcional)
tar -czf test_batch.tar.gz batches/modelo_dataset_*/

# Transferir por SSH, USB, etc.
scp test_batch.tar.gz usuario@gpu-machine:/path/to/work/
```

### Paso 3: Ejecutar en GPU 🎯
En la máquina con GPU:
```bash
# Descomprimir si es necesario
tar -xzf test_batch.tar.gz

# Generar script de GPU (si no tienes)
python model_tools_no_torch.py --action generate --output .

# Ejecutar inferencia
python gpu_inference.py tu_modelo.pt modelo_dataset_*/ results_output/
```

### Paso 4: Transferir Resultados 📊
```bash
# Comprimir resultados
tar -czf results.tar.gz results_output/

# Transferir de vuelta
scp results.tar.gz usuario@local-machine:/path/to/dental_ai/results/
```

### Paso 5: Procesar Local 📈
```bash
cd /Volumes/3TB/Ai/XRAY/Dist/dental_ai

# Descomprimir
tar -xzf results.tar.gz

# Procesar resultados
python workflow_no_torch.py
# Selecciona opción 5: Procesar resultados
```

## 📁 Estructura de Directorios

```
dental_ai/
├── models/                    # Modelos entrenados (.pt)
│   └── yolo_detect/
│       └── dental_dataset_best.pt
├── datasets/                  # Datasets organizados
│   └── detection_combined/
│       ├── data.yaml
│       ├── train/images/
│       ├── val/images/
│       └── test/images/
├── batches/                   # Lotes para GPU (generados)
│   └── modelo_dataset_timestamp/
├── results/                   # Resultados de GPU (recibidos)
│   └── resultado_timestamp/
├── workflow_no_torch.py       # 🎛️ Gestor principal
├── model_tools_no_torch.py    # 🔧 Herramientas específicas
└── test_model.py             # ⚠️ Requiere PyTorch (para GPU)
```

## 🚀 Inicio Rápido

1. **Verificar modelos disponibles:**
   ```bash
   python workflow_no_torch.py
   # Opción 1: Listar modelos
   ```

2. **Crear primer lote de prueba:**
   ```bash
   python workflow_no_torch.py
   # Opción 3: Crear lote
   # Selecciona modelo y dataset
   ```

3. **Seguir instrucciones** en el README.md generado

## 💡 Consejos

- **Lotes pequeños**: Usa 5-10 imágenes para pruebas rápidas
- **Compresión**: Usa `tar.gz` para transferencias más rápidas
- **Scripts**: El script `gpu_inference.py` se genera automáticamente
- **Resultados**: Guarda los resultados con nombres descriptivos

## 🆘 Solución de Problemas

### Error: "No se encontraron modelos"
```bash
# Verificar estructura
ls -la models/*/
# Los modelos deben estar en models/*/archivo.pt
```

### Error: "No se encontraron datasets" 
```bash
# Verificar estructura
ls -la datasets/*/data.yaml
# Debe existir data.yaml en cada dataset
```

### Error en GPU: "ModuleNotFoundError"
```bash
# En máquina GPU:
pip install ultralytics opencv-python
```

### Resultados vacíos
- Verificar que el modelo sea compatible con las imágenes
- Revisar el `data.yaml` para clases correctas
- Probar con imágenes del conjunto de entrenamiento

## 📞 Comandos de Emergencia

```bash
# Verificar todo rápidamente
find . -name "*.pt" -exec ls -lh {} \;  # Modelos
find datasets/ -name "data.yaml" -exec head -5 {} \;  # Datasets
ls -la batches/  # Lotes creados
ls -la results/  # Resultados recibidos
```

¡Listo para probar tu modelo sin PyTorch local! 🎉
