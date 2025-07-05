#!/usr/bin/env python3
"""
🩺 DIAGNÓSTICO COMPLETO - Dental AI Training
===========================================

Script para diagnosticar problemas en el sistema de entrenamiento.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    print(f"🐍 Python versión: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️ Advertencia: Se recomienda Python 3.8 o superior")
        return False
    else:
        print("✅ Versión de Python OK")
        return True

def check_package(package_name, import_name=None):
    """Verifica si un paquete está instalado."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NO INSTALADO")
        return False

def check_core_dependencies():
    """Verifica dependencias principales."""
    print("\n📦 DEPENDENCIAS PRINCIPALES:")
    print("-" * 30)
    
    deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("pyyaml", "yaml")
    ]
    
    results = {}
    for package, import_name in deps:
        results[package] = check_package(package, import_name)
    
    return results

def check_optional_dependencies():
    """Verifica dependencias opcionales."""
    print("\n📦 DEPENDENCIAS OPCIONALES:")
    print("-" * 30)
    
    deps = [
        ("detectron2", "detectron2"),
        ("tensorflow", "tensorflow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    results = {}
    for package, import_name in deps:
        results[package] = check_package(package, import_name)
    
    return results

def check_gpu_support():
    """Verifica soporte de GPU."""
    print("\n🚀 SOPORTE DE GPU:")
    print("-" * 20)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA disponible: {torch.version.cuda}")
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ CUDA no disponible - usando CPU")
            
        # MPS para macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Metal Performance Shaders (MPS) disponible")
            
    except ImportError:
        print("❌ No se puede verificar GPU - PyTorch no instalado")

def check_training_scripts():
    """Verifica que los scripts de entrenamiento existan."""
    print("\n📝 SCRIPTS DE ENTRENAMIENTO:")
    print("-" * 30)
    
    script_dir = Path(__file__).parent
    scripts = [
        "train_cls_dental_dataset.py",
        "train_seg_dental_dataset.py", 
        "train_yolo_dental_dataset.py",
        "train_master.py"
    ]
    
    for script in scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - NO ENCONTRADO")

def check_datasets():
    """Verifica que los datasets existan."""
    print("\n📊 DATASETS:")
    print("-" * 15)
    
    base_path = Path(__file__).parent.parent / "datasets"
    
    datasets = [
        ("YOLO", "detection/dental_dataset/data.yaml"),
        ("COCO", "segmentation/dental_dataset/train/annotations.json"),
        ("Clasificación", "classification/dental_dataset/train")
    ]
    
    for name, path in datasets:
        full_path = base_path / path
        if full_path.exists():
            print(f"✅ {name}: {full_path}")
        else:
            print(f"❌ {name}: {full_path} - NO ENCONTRADO")

def run_syntax_check():
    """Verifica sintaxis de scripts Python."""
    print("\n🔍 VERIFICACIÓN DE SINTAXIS:")
    print("-" * 30)
    
    script_dir = Path(__file__).parent
    python_scripts = script_dir.glob("*.py")
    
    for script in python_scripts:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {script.name}")
            else:
                print(f"❌ {script.name}: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"❌ {script.name}: Error verificando - {e}")

def generate_installation_commands():
    """Genera comandos de instalación para dependencias faltantes."""
    print("\n🛠️ COMANDOS DE INSTALACIÓN:")
    print("-" * 30)
    
    print("# Dependencias principales:")
    print("pip install torch torchvision ultralytics opencv-python numpy pillow pyyaml")
    
    print("\n# Detectron2 (para segmentación):")
    print("# GPU:")
    print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html")
    print("# CPU:")
    print("pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    
    print("\n# Dependencias opcionales:")
    print("pip install matplotlib seaborn tensorflow")

def main():
    print("🦷 DIAGNÓSTICO COMPLETO - Dental AI Training")
    print("=" * 50)
    
    # Verificaciones principales
    check_python_version()
    check_core_dependencies()
    check_optional_dependencies()
    check_gpu_support()
    check_training_scripts()
    check_datasets()
    run_syntax_check()
    
    # Comandos de instalación
    generate_installation_commands()
    
    print("\n" + "=" * 50)
    print("✅ Diagnóstico completado")
    print("💡 Revisa los elementos marcados con ❌ para resolver problemas")

if __name__ == "__main__":
    main()
