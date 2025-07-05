#!/usr/bin/env python3
"""
🔍 Diagnóstico simple del problema
"""

from pathlib import Path

def verificar_estructura():
    """Verificar estructura existente."""
    print("🔍 VERIFICANDO ESTRUCTURA ACTUAL")
    print("=" * 40)
    
    # Verificar directorio de salida
    output_path = Path("Dist/dental_ai")
    print(f"📂 {output_path}: {output_path.exists()}")
    
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir():
                print(f"   📁 {item.name}/")
                for subitem in item.iterdir():
                    print(f"      • {subitem.name}")
            else:
                print(f"   📄 {item.name}")
    
    # Verificar específicamente lo que busca _generate_training_scripts
    datasets_dir = output_path / "datasets"
    if datasets_dir.exists():
        yolo_dir = datasets_dir / "yolo"
        coco_dir = datasets_dir / "coco"
        unet_dir = datasets_dir / "unet"
        
        print(f"\n🔄 VERIFICANDO CONDICIONES PARA GENERAR SCRIPTS:")
        print(f"   datasets/yolo/ existe: {yolo_dir.exists()}")
        print(f"   datasets/coco/ existe: {coco_dir.exists()}")
        print(f"   datasets/unet/ existe: {unet_dir.exists()}")
        
        if not any([yolo_dir.exists(), coco_dir.exists(), unet_dir.exists()]):
            print("❌ PROBLEMA ENCONTRADO: No hay datasets convertidos")
            print("💡 Necesitas ejecutar la conversión de formatos antes de generar scripts")
    else:
        print("❌ PROBLEMA ENCONTRADO: No existe datasets/")
        print("💡 Necesitas ejecutar el workflow de conversión primero")

def crear_ejemplo_estructura():
    """Crear estructura de ejemplo para probar."""
    print(f"\n🧪 CREANDO ESTRUCTURA DE EJEMPLO...")
    
    # Crear estructura mínima
    example_path = Path("Dist/dental_ai_ejemplo")
    scripts_dir = example_path / "scripts"
    datasets_dir = example_path / "datasets"
    yolo_dir = datasets_dir / "yolo"
    
    # Crear directorios
    yolo_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)
    
    # Crear archivo data.yaml de ejemplo
    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text("""
# Dental AI Dataset Configuration
path: .
train: train
val: val
test: test

names:
  0: caries
  1: tooth
  2: implant
""")
    
    print(f"✅ Estructura de ejemplo creada en: {example_path}")
    
    # Simular generación de scripts
    script_content = '''#!/usr/bin/env python3
"""
🎯 Script de entrenamiento YOLO para datasets dentales
"""

import torch
from ultralytics import YOLO

def train_yolo_model():
    """Entrenar modelo YOLO."""
    # Cargar modelo pre-entrenado
    model = YOLO('yolov8n.pt')
    
    # Configurar entrenamiento
    results = model.train(
        data='../datasets/yolo/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='auto'
    )
    
    return results

if __name__ == "__main__":
    train_yolo_model()
'''
    
    script_path = scripts_dir / "train_yolo.py"
    script_path.write_text(script_content)
    
    print(f"✅ Script de ejemplo creado: {script_path}")
    
    return example_path

if __name__ == "__main__":
    verificar_estructura()
    crear_ejemplo_estructura()
    
    print(f"\n🎯 SOLUCIÓN AL PROBLEMA:")
    print("=" * 30)
    print("El problema es que la opción 7 requiere que antes hayas:")
    print("1. Escaneado datasets (opción 1)")
    print("2. Seleccionado datasets (opción 3)")
    print("3. Convertido a algún formato (opción 4)")
    print()
    print("Solo entonces se crearán las carpetas datasets/yolo/, etc.")
    print("Y la opción 7 podrá generar los scripts.")
    print()
    print("🚀 RECOMENDACIÓN: Usa la opción 8 (Workflow completo)")
    print("   Esto ejecuta todo el proceso automáticamente.")
