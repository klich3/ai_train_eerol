"""
🦷 Script Templates Generator
Generador de templates para scripts de entrenamiento y APIs
"""

import os
import stat
from pathlib import Path
from datetime import datetime


class ScriptTemplateGenerator:
    """Generador de templates para scripts de entrenamiento."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def create_yolo_training_script(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento específico para YOLO."""
        # Colocar el script en la carpeta del dataset para usar rutas relativas
        dataset_dir = self.output_path / 'datasets' / 'detection_combined'
        script_file = dataset_dir / f'train_{dataset_name}.sh'
        
        # También crear una copia en training/ para compatibilidad
        training_dir = self.output_path / 'training'
        backup_script_file = training_dir / f'train_{dataset_name}.sh'
        
        script_content = f"""#!/bin/bash
# 🦷 Entrenamiento específico para {dataset_name}
# Este script debe ejecutarse desde la carpeta del dataset

echo "🦷 Iniciando entrenamiento para {dataset_name}..."

# Obtener directorio actual (donde está el dataset y el data.yaml)
DATASET_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
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
OUTPUT_DIR="$DATASET_DIR/logs/training_${{TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

echo "📁 Guardando resultados en: $OUTPUT_DIR"

# Entrenamiento YOLO
echo "🚀 Iniciando entrenamiento YOLO..."
yolo detect train \\
    model=$MODEL \\
    data="$TEMP_DATA_YAML" \\
    epochs=$EPOCHS \\
    batch=$BATCH \\
    imgsz=$IMG_SIZE \\
    project="$OUTPUT_DIR" \\
    name="{dataset_name}" \\
    save_period=10 \\
    patience=20 \\
    device=0 \\
    workers=8 \\
    cache=True

echo "✅ Entrenamiento completado"
echo "📁 Modelo guardado en: $OUTPUT_DIR/{dataset_name}/weights/"
echo "📊 Métricas en: $OUTPUT_DIR/{dataset_name}/"

# Copiar mejor modelo a directorio de modelos
MODEL_DIR="$DATASET_DIR/../../models/yolo_detect"
mkdir -p "$MODEL_DIR"
if [[ -f "$OUTPUT_DIR/{dataset_name}/weights/best.pt" ]]; then
    cp "$OUTPUT_DIR/{dataset_name}/weights/best.pt" "$MODEL_DIR/{dataset_name}_best.pt"
    echo "📦 Modelo copiado a: $MODEL_DIR/{dataset_name}_best.pt"
else
    echo "⚠️ No se encontró el modelo entrenado en: $OUTPUT_DIR/{dataset_name}/weights/best.pt"
fi

# Limpiar archivo temporal
rm -f "$TEMP_DATA_YAML"
echo "🧹 Archivo temporal eliminado"

echo "🎉 Proceso completado!"
echo "💡 Para ejecutar nuevamente:"
echo "   cd $DATASET_DIR"
echo "   ./train_{dataset_name}.sh"
"""
        
        # Crear el script en la carpeta del dataset (principal)
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        # Crear también una copia en training/ para compatibilidad
        with open(backup_script_file, 'w') as f:
            f.write(script_content)
        backup_script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento YOLO creado en: {script_file}")
        print(f"📝 Copia de respaldo creada en: {backup_script_file}")
        print(f"💡 Para ejecutar: cd {dataset_dir} && ./train_{dataset_name}.sh")
    
    def create_segmentation_training_script(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento para segmentación."""
        training_dir = self.output_path / 'training'
        script_file = training_dir / f'train_seg_{dataset_name}.py'
        
        script_content = '''#!/usr/bin/env python3
# 🦷 Entrenamiento de segmentación para {dataset_name}

import os
import sys
import torch

# Verificar que detectron2 esté instalado
try:
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2 import model_zoo
    from detectron2.utils.logger import setup_logger
    DETECTRON2_AVAILABLE = True
except ImportError as e:
    print("❌ Error: Detectron2 no está instalado")
    print("💡 Para instalar detectron2, ejecuta:")
    print("   python install_detectron2.py")
    print("   o manualmente:")
    print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html")
    print(f"📝 Error específico: {{e}}")
    DETECTRON2_AVAILABLE = False

def setup_config():
    """Configuración para entrenamiento de segmentación."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Configuración del dataset
    cfg.DATASETS.TRAIN = ("{dataset_name}_train",)
    cfg.DATASETS.TEST = ("{dataset_name}_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Configuración de entrenamiento
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (2000,)
    cfg.SOLVER.GAMMA = 0.1
    
    # Configuración del modelo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("{dataset_name}_train").thing_classes)
    
    # Directorio de salida
    cfg.OUTPUT_DIR = f"./logs/{dataset_name}_segmentation"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def register_datasets():
    """Registra los datasets COCO."""
    # Usar rutas relativas desde el directorio de ejecución
    dataset_path = "../datasets/segmentation_coco"
    
    print("📁 Dataset COCO: " + dataset_path)
    
    if not os.path.exists(dataset_path):
        print("❌ Error: No se encontró el dataset: " + dataset_path)
        print("💡 Ejecute desde training/ y asegúrese de que existe:")
        print("   ../datasets/segmentation_coco/")
        return
    
    # Verificar estructura del dataset
    if os.path.exists(os.path.join(dataset_path, "annotations")):
        # Estructura: dataset/annotations/, dataset/images/
        register_coco_instances(
            "{dataset_name}_train",
            {{}},
            os.path.join(dataset_path, "annotations", "instances_train.json"),
            os.path.join(dataset_path, "images")
        )
        register_coco_instances(
            "{dataset_name}_val",
            {{}},
            os.path.join(dataset_path, "annotations", "instances_val.json"),
            os.path.join(dataset_path, "images")
        )
    else:
        # Estructura: dataset/train/, dataset/val/, dataset/test/
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                register_coco_instances(
                    "{dataset_name}_" + split,
                    {{}},
                    os.path.join(split_path, "annotations.json"),
                    split_path
                )

def main():
    if not DETECTRON2_AVAILABLE:
        print("🚫 No se puede entrenar segmentación sin detectron2")
        print("💡 Instala detectron2 primero con: python install_detectron2.py")
        sys.exit(1)
        
    setup_logger()
    print("🦷 Iniciando entrenamiento de segmentación para {dataset_name}")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {{torch.cuda.get_device_name()}}")
    else:
        print("⚠️ Usando CPU - el entrenamiento será más lento")
    
    # Registrar datasets
    register_datasets()
    
    # Configurar entrenamiento
    cfg = setup_config()
    
    # Crear trainer y entrenar
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("✅ Entrenamiento completado")
    print(f"📁 Modelo guardado en: {{cfg.OUTPUT_DIR}}")

if __name__ == "__main__":
    main()
'''.format(dataset_name=dataset_name, target_type=target_type)
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento de segmentación creado: {script_file}")
    
    def create_classification_training_script(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento para clasificación."""
        training_dir = self.output_path / 'training'
        script_file = training_dir / f'train_cls_{dataset_name}.py'
        
        script_content = f'''#!/usr/bin/env python3
# 🦷 Entrenamiento de clasificación para {dataset_name}

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import sys
from datetime import datetime

def create_data_loaders(data_dir, batch_size=32):
    """Crea data loaders para entrenamiento."""
    
    # Verificar que existan los directorios
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Si no existen train/val directamente, buscar en subdirectorios
    if not os.path.exists(train_dir):
        # Buscar el primer subdirectorio que contenga train/val
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for subdir in subdirs:
            potential_train = os.path.join(data_dir, subdir, 'train')
            potential_val = os.path.join(data_dir, subdir, 'val')
            if os.path.exists(potential_train) and os.path.exists(potential_val):
                train_dir = potential_train
                val_dir = potential_val
                print(f"📁 Usando dataset en: {{subdir}}")
                break
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Directorio de entrenamiento no encontrado: {{train_dir}}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"❌ Directorio de validación no encontrado: {{val_dir}}")
    
    print(f"📂 Train: {{train_dir}}")
    print(f"📂 Val: {{val_dir}}")
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes):
    """Crea modelo ResNet para clasificación."""
    model = models.resnet50(pretrained=True)
    
    # Congelar capas iniciales
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar clasificador final
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Entrena el modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Directorio de salida
    output_dir = f"./logs/{dataset_name}_classification"
    os.makedirs(output_dir, exist_ok=True)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {{epoch+1}}/{{num_epochs}}")
        print("-" * 20)
        
        # Entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {{epoch_loss:.4f}} Acc: {{epoch_acc:.4f}}")
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {{val_loss:.4f}} Acc: {{val_acc:.4f}}")
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, 'best_model.pth'))
        
        scheduler.step()
        print()
    
    print(f"Mejor precisión de validación: {{best_acc:.4f}}")
    return model

def main():
    print("🦷 Iniciando entrenamiento de clasificación para {dataset_name}")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {{torch.cuda.get_device_name()}}")
    else:
        print("⚠️ Usando CPU - el entrenamiento será más lento")
    
    # Usar rutas relativas desde el directorio de ejecución
    data_dir = "../datasets/classification"
    
    print(f"📁 Directorio de datos: {{data_dir}}")
    
    # Verificar que existe el directorio
    if not os.path.exists(data_dir):
        print(f"❌ Error: No se encontró el directorio de datos: {{data_dir}}")
        print("💡 Ejecute desde training/ y asegúrese de que existe:")
        print("   ../datasets/classification/{{train,val}}/")
        return
    
    batch_size = 32
    num_epochs = 50
    
    try:
        # Crear data loaders
        train_loader, val_loader, classes = create_data_loaders(data_dir, batch_size)
        print(f"Clases encontradas: {{classes}}")
        print(f"Número de clases: {{len(classes)}}")
        
        # Crear modelo
        model = create_model(len(classes))
        
        # Entrenar
        trained_model = train_model(model, train_loader, val_loader, num_epochs)
        
        print("✅ Entrenamiento completado")
        print(f"📁 Modelo guardado en: ./logs/{dataset_name}_classification/")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {{e}}")
        print("💡 Verifica que:")
        print("   - El dataset existe en la ruta especificada")
        print("   - Las carpetas train/ y val/ contienen subdirectorios de clases")
        print("   - Hay suficientes imágenes en cada clase")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento de clasificación creado: {script_file}")
    
    def create_api_template(self):
        """Crea plantilla de API para inferencia."""
        api_dir = self.output_path / 'api'
        api_dir.mkdir(parents=True, exist_ok=True)
        
        # API principal
        main_api_file = api_dir / 'main.py'
        api_content = '''#!/usr/bin/env python3
# 🦷 API de Inferencia Dental AI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import uvicorn

app = FastAPI(
    title="🦷 Dental AI API",
    description="API para análisis dental con deep learning",
    version="1.0.0"
)

# Configuración global
MODELS_DIR = Path("../models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache de modelos cargados
loaded_models = {}

@app.get("/")
async def root():
    return {
        "message": "🦷 Dental AI API",
        "version": "1.0.0",
        "status": "active",
        "device": str(DEVICE)
    }

@app.get("/models")
async def list_models():
    """Lista modelos disponibles."""
    models = {
        "detection": list((MODELS_DIR / "yolo_detect").glob("*.pt")) if (MODELS_DIR / "yolo_detect").exists() else [],
        "segmentation": list((MODELS_DIR / "yolo_segment").glob("*.pt")) if (MODELS_DIR / "yolo_segment").exists() else [],
        "classification": list((MODELS_DIR / "cnn_classifier").glob("*.pth")) if (MODELS_DIR / "cnn_classifier").exists() else []
    }
    
    return {
        "available_models": {k: [str(m.name) for m in v] for k, v in models.items()},
        "total_models": sum(len(v) for v in models.values())
    }

@app.post("/predict/detection")
async def predict_detection(
    file: UploadFile = File(...),
    model_name: str = "dental_dataset_best.pt",
    confidence: float = 0.25
):
    """Detección de estructuras dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia YOLO
        # Por ahora retornamos un ejemplo
        
        return {
            "predictions": [
                {
                    "class": "tooth",
                    "confidence": 0.89,
                    "bbox": [100, 150, 200, 250]
                },
                {
                    "class": "caries",
                    "confidence": 0.67,
                    "bbox": [180, 200, 220, 240]
                }
            ],
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.45s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...),
    model_name: str = "dental_dataset_best.pth"
):
    """Clasificación de patologías dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia CNN
        # Por ahora retornamos un ejemplo
        
        return {
            "prediction": {
                "class": "caries",
                "confidence": 0.87,
                "probabilities": {
                    "normal": 0.13,
                    "caries": 0.87,
                    "filling": 0.00
                }
            },
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.23s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificación de salud del API."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": len(loaded_models)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
'''
        
        with open(main_api_file, 'w') as f:
            f.write(api_content)
        
        # Archivo de requirements para la API
        api_requirements = api_dir / 'requirements.txt'
        with open(api_requirements, 'w') as f:
            f.write('''fastapi>=0.100.0
uvicorn[standard]>=0.22.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=9.5.0
python-multipart>=0.0.6
''')
        
        print(f"📝 API template creada en: {api_dir}")
        print(f"🚀 Para ejecutar: cd {api_dir.parent.name}/api && python main.py")
