# 🔧 EEROL - Universal Dataset Management Tool

**EEROL** es una herramienta universal para la gestión de datasets de visión por computador. Permite escanear, analizar, convertir, dividir y entrenar modelos con datasets en diferentes formatos.

## ✨ Características

- 🔍 **Escaneo automático** de datasets en cualquier directorio
- 📊 **Análisis detallado** de estructura, formatos y categorías
- 🔄 **Conversión** entre formatos (YOLO ↔ COCO ↔ Pascal VOC)
- ✂️ **División personalizada** con proporciones configurables (train/val/test)
- 🚀 **Generación automática** de scripts de entrenamiento
- 👁️ **Previsualización** de anotaciones sobre imágenes
- 🧹 **Limpieza automática** de archivos innecesarios
- 🎯 **Soporte múltiple** para YOLO, COCO, PyTorch, TensorFlow, U-Net

## 🚀 Instalación

1. **Clonar o descargar** este repositorio
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Hacer ejecutable** (en Linux/macOS):
   ```bash
   chmod +x eerol.py
   ```

## 📋 Uso

### Modo Interactivo

```bash
python eerol.py
```

### Línea de Comandos

#### Escanear datasets

```bash
python eerol.py scan --path /ruta/a/datasets
python eerol.py scan  # Usa directorio actual o HOME
```

#### Convertir formato

```bash
python eerol.py convert --input-path /ruta/dataset --format yolo --name mi_dataset
python eerol.py convert --input-path /ruta/dataset --format coco
```

#### Previsualizar anotaciones

```bash
python eerol.py preview --image imagen.jpg --annotation annotation.txt --format yolo
python eerol.py preview --image imagen.jpg --annotation annotation.xml --format pascal_voc
```

#### Dividir dataset

```bash
python eerol.py split --input-path /ruta/dataset --train-ratio 0.7 --val-ratio 0.3
python eerol.py split --input-path /ruta/dataset --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
```

#### Listar datasets de entrenamiento

```bash
python eerol.py list
```

#### Entrenar modelo

```bash
python eerol.py train --dataset mi_dataset
python eerol.py train  # Selección interactiva
```

#### Limpiar archivos

```bash
python eerol.py clean
```

## 📁 Estructura de Salida

EEROL genera los datasets en la carpeta `Train/` con la siguiente estructura:

```
Train/
├── mi_dataset_yolo/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/           # Opcional
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml       # Configuración YOLO
│   ├── split_info.yaml # Información de división
│   └── train.py        # Script de entrenamiento
```

## 🎯 Formatos Soportados

### Entrada (Detección Automática)

- **YOLO**: `.txt` + `data.yaml`
- **COCO**: `.json` con estructura estándar
- **Pascal VOC**: `.xml` con anotaciones

### Salida (Conversión)

- **YOLO**: Estructura estándar con `data.yaml`
- **COCO**: JSON con imágenes y anotaciones
- **Pascal VOC**: XML individual por imagen

## 🚀 Scripts de Entrenamiento

EEROL genera automáticamente scripts de entrenamiento optimizados:

- **YOLOv8**: Usando ultralytics
- **COCO**: Base para Detectron2/MMDetection
- **PyTorch**: Plantilla personalizable
- **TensorFlow**: Base para TF Object Detection API
- **U-Net**: Para segmentación semántica

## 🔧 Configuración

EEROL crea automáticamente:

- `~/.eerol/config.yaml`: Configuración global
- `Train/`: Directorio de datasets generados
- `Results/`: Directorio de resultados
- `Backups/`: Directorio de respaldos

## 📊 Ejemplo de Uso Completo

1. **Escanear** datasets existentes:

   ```bash
   python eerol.py scan --path ~/datasets
   ```

2. **Convertir** a YOLO:

   ```bash
   python eerol.py convert --input-path ~/datasets/mi_coco_dataset --format yolo --name converted_yolo
   ```

3. **Dividir** con proporciones personalizadas:

   ```bash
   python eerol.py split --input-path Train/converted_yolo --train-ratio 0.8 --val-ratio 0.2 --name final_dataset
   ```

4. **Entrenar** el modelo:
   ```bash
   python eerol.py train --dataset final_dataset
   ```

## 🛠️ Personalización

### Agregar Nuevos Formatos

Edita `eerol/dataset_converter.py` para agregar nuevos formatos de conversión.

### Personalizar Scripts de Entrenamiento

Modifica `eerol/script_generator.py` para agregar nuevos frameworks o personalizar parámetros.

### Agregar Nuevas Validaciones

Extiende `eerol/utils.py` para agregar validaciones específicas de formato.

## 🧹 Limpieza

EEROL puede limpiar automáticamente:

- Archivos `__pycache__`
- Archivos temporales
- Archivos obsoletos del proyecto anterior
- Caches de frameworks

## ⚠️ Notas Importantes

- **Respaldo**: EEROL siempre preserva los datasets originales
- **Dependencias**: Los frameworks de ML se instalan bajo demanda
- **Memoria**: Para datasets grandes, considera usar SSD
- **GPU**: Los scripts detectan automáticamente disponibilidad de GPU

## 🤝 Contribuciones

Este es un proyecto refactorizado de una herramienta específica para datasets dentales, ahora convertida en una herramienta universal. Las contribuciones son bienvenidas.

## 📄 Licencia

Proyecto de código abierto. Ver archivo de licencia para más detalles.

---

**¡EEROL hace que la gestión de datasets de visión por computador sea simple y eficiente!** 🚀
