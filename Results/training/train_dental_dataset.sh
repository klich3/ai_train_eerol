#!/bin/bash
# 🦷 Entrenamiento específico para dental_dataset
# Este script debe ejecutarse desde la carpeta del dataset

echo "🦷 Iniciando entrenamiento para dental_dataset..."

# Obtener directorio actual (donde está el dataset y el data.yaml)
DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Directorio del dataset: $DATASET_DIR"

# Verificar que estamos en la carpeta correcta
if [[ ! -f "$DATASET_DIR/data.yaml" ]]; then
    echo "❌ Error: No se encontró data.yaml en el directorio actual"
    echo "💡 Este script debe ejecutarse desde la carpeta del dataset"
    echo "💡 Estructura esperada:"
    echo "   detection_combined/"
    echo "   ├── data.yaml"
    echo "   ├── train/images/"
    echo "   ├── val/images/"
    echo "   └── test/images/"
    exit 1
fi

# Verificar estructura del dataset
for DIR in "train/images" "val/images" "test/images"; do
    if [[ ! -d "$DATASET_DIR/$DIR" ]]; then
        echo "❌ Error: No se encontró directorio $DIR"
        exit 1
    fi
done

echo "✅ Estructura del dataset verificada"

# Configuración específica
MODEL="yolov8n.pt"
EPOCHS=100
BATCH=16
IMG_SIZE=640

# Verificar y corregir data.yaml para usar rutas relativas
DATA_YAML="$DATASET_DIR/data.yaml"
echo "🔧 Verificando configuración en: $DATA_YAML"

# Crear una copia temporal del data.yaml con rutas relativas correctas
TEMP_DATA_YAML="$DATASET_DIR/data_temp.yaml"

# Leer el data.yaml original y corregir las rutas
python3 << 'EOF'
import yaml
import os

# Leer el data.yaml original
with open('data.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Corregir las rutas para que sean relativas al directorio actual
data['path'] = '.'  # Directorio actual
data['train'] = 'train/images'
data['val'] = 'val/images' 
data['test'] = 'test/images'

# Guardar el data.yaml temporal
with open('data_temp.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("✅ data.yaml temporal creado con rutas relativas")
EOF

echo "📝 Contenido del data.yaml temporal:"
cat "$TEMP_DATA_YAML"

# Crear directorio de salida para logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$DATASET_DIR/logs/training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📁 Guardando resultados en: $OUTPUT_DIR"

# Entrenamiento YOLO
echo "🚀 Iniciando entrenamiento YOLO..."
yolo detect train \
    model=$MODEL \
    data="$TEMP_DATA_YAML" \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMG_SIZE \
    project="$OUTPUT_DIR" \
    name="dental_dataset" \
    save_period=10 \
    patience=20 \
    device=0 \
    workers=8 \
    cache=True

echo "✅ Entrenamiento completado"
echo "📁 Modelo guardado en: $OUTPUT_DIR/dental_dataset/weights/"
echo "📊 Métricas en: $OUTPUT_DIR/dental_dataset/"

# Copiar mejor modelo a directorio de modelos
MODEL_DIR="$DATASET_DIR/../../models/yolo_detect"
mkdir -p "$MODEL_DIR"
if [[ -f "$OUTPUT_DIR/dental_dataset/weights/best.pt" ]]; then
    cp "$OUTPUT_DIR/dental_dataset/weights/best.pt" "$MODEL_DIR/dental_dataset_best.pt"
    echo "📦 Modelo copiado a: $MODEL_DIR/dental_dataset_best.pt"
else
    echo "⚠️ No se encontró el modelo entrenado en: $OUTPUT_DIR/dental_dataset/weights/best.pt"
fi

# Limpiar archivo temporal
rm -f "$TEMP_DATA_YAML"
echo "🧹 Archivo temporal eliminado"

echo "🎉 Proceso completado!"
echo "💡 Para ejecutar nuevamente:"
echo "   cd $DATASET_DIR"
echo "   ./train_dental_dataset.sh"
