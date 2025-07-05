#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE RUTAS - Específico para problemas de paths
============================================================
"""

import os
import sys
from pathlib import Path

def check_current_directory():
    """Verifica el directorio actual."""
    cwd = os.getcwd()
    print(f"📁 Directorio actual: {cwd}")
    
    # Verificar si estamos en training/
    if cwd.endswith('/training'):
        print("✅ Ejecutándose desde training/")
        base_dir = Path(cwd).parent
        print(f"📁 Directorio base (dental_ai): {base_dir}")
        return base_dir
    else:
        print("⚠️ No se está ejecutando desde training/")
        return None

def check_relative_paths(base_dir):
    """Verifica las rutas relativas desde training/."""
    if base_dir is None:
        print("❌ No se puede verificar sin directorio base")
        return
    
    training_dir = base_dir / "training"
    print(f"\n🔍 Verificando rutas desde: {training_dir}")
    
    # Simular estar en training/
    os.chdir(training_dir)
    print(f"📁 Cambiado a: {os.getcwd()}")
    
    # Verificar rutas relativas
    paths_to_check = [
        "../datasets/detection_combined/data.yaml",
        "../datasets/classification/train",
        "../datasets/classification/val",
        "../datasets/segmentation_coco/annotations"
    ]
    
    for path in paths_to_check:
        full_path = Path(path).resolve()
        exists = Path(path).exists()
        print(f"{'✅' if exists else '❌'} {path} → {full_path}")

def check_yaml_paths():
    """Verifica las rutas en el data.yaml."""
    yaml_path = "../datasets/detection_combined/data.yaml"
    
    if os.path.exists(yaml_path):
        print(f"\n📄 Contenido de {yaml_path}:")
        with open(yaml_path, 'r') as f:
            content = f.read()
            print(content)
    else:
        print(f"\n❌ No se encontró {yaml_path}")

def main():
    print("🔍 DIAGNÓSTICO DE RUTAS")
    print("=" * 50)
    
    original_cwd = os.getcwd()
    print(f"📁 Directorio inicial: {original_cwd}")
    
    # Si no estamos en training, ir allí
    if not original_cwd.endswith('/training'):
        training_path = Path(original_cwd) / "Dist/dental_ai/training"
        if training_path.exists():
            print(f"📁 Cambiando a: {training_path}")
            os.chdir(training_path)
        else:
            print(f"❌ No se encontró: {training_path}")
            return
    
    base_dir = check_current_directory()
    check_relative_paths(base_dir)
    check_yaml_paths()
    
    # Restaurar directorio original
    os.chdir(original_cwd)

if __name__ == "__main__":
    main()
