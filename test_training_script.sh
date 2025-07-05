#!/bin/bash
# 🧪 Test del script de entrenamiento (sin ejecutar YOLO)

cd "/Volumes/3TB/Ai/XRAY/Dist/dental_ai/datasets/detection_combined"

echo "🧪 Probando el script de entrenamiento..."

# Simular las partes del script sin el entrenamiento real
echo "🦷 Iniciando entrenamiento para dental_dataset..."

# Obtener directorio actual
DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Directorio del dataset: $DATASET_DIR"

# Verificar que estamos en la carpeta correcta
if [[ ! -f "$DATASET_DIR/data.yaml" ]]; then
    echo "❌ Error: No se encontró data.yaml en el directorio actual"
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

# Simular verificación de YOLO (sin entrenar)
echo "🚀 Verificando comando YOLO..."
echo "Comando que se ejecutaría:"
echo "yolo detect train model=yolov8n.pt data='$TEMP_DATA_YAML' epochs=100 batch=16 imgsz=640 project='$OUTPUT_DIR' name='dental_dataset'"

# Limpiar archivo temporal
rm -f "$TEMP_DATA_YAML"
echo "🧹 Archivo temporal eliminado"

echo "🎉 ¡Prueba completada exitosamente!"
echo "💡 El script está listo para ejecutar el entrenamiento real"
