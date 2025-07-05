#!/usr/bin/env python3
"""
🚀 Script de entrenamiento YOLO para dental_dataset con rutas dinámicas
"""

import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

def generate_dynamic_config():
    """Genera configuración YOLO con rutas absolutas dinámicas."""
    
    # Calcular rutas dinámicamente
    script_dir = Path(__file__).parent.resolve()
    dental_ai_dir = script_dir.parent
    dataset_dir = dental_ai_dir / "datasets" / "detection_combined"
    
    print(f"📁 Script dir: {script_dir}")
    print(f"📁 Dental AI dir: {dental_ai_dir}")
    print(f"📁 Dataset dir: {dataset_dir}")
    
    # Verificar que existe el dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_dir}")
    
    # Leer configuración original
    original_config = dataset_dir / "data.yaml"
    config = {
        'nc': 6,
        'names': {i: f'class_{i}' for i in range(6)}
    }
    
    if original_config.exists():
        with open(original_config, 'r') as f:
            original = yaml.safe_load(f)
            if 'nc' in original:
                config['nc'] = original['nc']
            if 'names' in original:
                config['names'] = original['names']
    
    # Configurar rutas absolutas
    config.update({
        'path': str(dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    })
    
    # Verificar splits
    for split in ['train', 'val', 'test']:
        split_path = dataset_dir / split / 'images'
        if split_path.exists():
            img_count = len(list(split_path.glob('*.jpg')) + 
                          list(split_path.glob('*.jpeg')) + 
                          list(split_path.glob('*.png')))
            print(f"✅ {split}: {img_count} imágenes")
        else:
            print(f"⚠️ {split}: No encontrado")
    
    # Guardar configuración temporal
    temp_config = script_dir / "temp_yolo_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Config temporal: {temp_config}")
    return str(temp_config)

def main():
    print("🚀 Entrenamiento YOLO - Dental Dataset")
    print("=" * 50)
    
    try:
        # Generar configuración dinámica
        config_file = generate_dynamic_config()
        
        # Configuración de entrenamiento
        model_name = "yolov8n.pt"
        epochs = 100
        batch_size = 16
        img_size = 640
        
        print(f"🤖 Modelo: {model_name}")
        print(f"📊 Épocas: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🖼️ Tamaño imagen: {img_size}")
        
        # Crear modelo
        model = YOLO(model_name)
        
        # Directorio de salida
        script_dir = Path(__file__).parent
        output_dir = script_dir / "logs" / "dental_yolo_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Resultados en: {output_dir}")
        
        # Entrenar
        results = model.train(
            data=config_file,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(output_dir),
            name="dental_dataset",
            save_period=10,
            patience=20,
            device=0,
            workers=8,
            cache=True
        )
        
        print("✅ Entrenamiento completado")
        
        # Copiar mejor modelo
        dental_ai_dir = script_dir.parent
        models_dir = dental_ai_dir / "models" / "yolo_detect"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        best_model = output_dir / "dental_dataset" / "weights" / "best.pt"
        if best_model.exists():
            target = models_dir / "dental_dataset_best.pt"
            import shutil
            shutil.copy(best_model, target)
            print(f"📦 Modelo copiado a: {target}")
        
        # Limpiar archivo temporal
        temp_config = Path(config_file)
        if temp_config.exists():
            temp_config.unlink()
            print("🧹 Archivo temporal eliminado")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
