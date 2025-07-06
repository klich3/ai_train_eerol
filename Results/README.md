# 🦷 Dental AI - Sistema de Análisis Dental con IA

Proyecto completo para análisis dental usando deep learning, generado automáticamente por DataWorkflowManager.

## 📁 Estructura del Proyecto

```
dental-ai/
├── datasets/           # Datasets procesados y listos para entrenamiento
│   ├── detection_combined/     # Datasets YOLO fusionados para detección
│   ├── segmentation_coco/      # Datasets COCO para segmentación
│   ├── segmentation_bitmap/    # Máscaras para U-Net
│   └── classification/         # Datasets para clasificación
├── models/             # Modelos entrenados
│   ├── yolo_detect/           # Modelos YOLO para detección
│   ├── yolo_segment/          # Modelos YOLO para segmentación
│   ├── unet_teeth/            # Modelos U-Net para dientes
│   └── cnn_classifier/        # Clasificadores CNN
├── training/           # Scripts y configuraciones de entrenamiento
│   ├── scripts/               # Scripts de entrenamiento automatizados
│   ├── configs/               # Configuraciones específicas
│   └── logs/                  # Logs de entrenamiento
├── api/                # API REST para inferencia
├── docs/               # Documentación adicional
└── README.md          # Este archivo
```

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenamiento de Modelos
```bash
cd training/scripts
bash train_[nombre_dataset].sh
```

### 3. Uso de la API
```bash
cd api
python main.py
```

## 📊 Datasets Disponibles

Los datasets están organizados por tipo de tarea:

- **Detección**: Datasets YOLO para detectar estructuras dentales
- **Segmentación**: Datasets COCO y máscaras bitmap para segmentación precisa
- **Clasificación**: Datasets organizados por carpetas para clasificación de patologías

## 🔧 Configuración

Todos los parámetros de entrenamiento están en `training/configs/`.

## 📝 Logs y Monitoreo

Los logs de entrenamiento se guardan en `training/logs/` con timestamps.

## 🛡️ Protección de Datos

Este proyecto utiliza un sistema de seguridad que:
- ✅ NUNCA modifica los datos originales
- ✅ Crea copias de solo lectura
- ✅ Verifica la integridad de los archivos copiados
- ✅ Mantiene logs completos de todas las operaciones

## 📈 Desarrollo

Generado automáticamente el 2025-07-06 00:38:00 por DataWorkflowManager.

Para regenerar o actualizar datasets, utiliza el DataWorkflowManager en el directorio padre.
