#!/usr/bin/env python3
"""
🔧 Fix Data YAML
Corrige el data.yaml para usar rutas relativas
"""

import yaml
import os
from pathlib import Path

def fix_data_yaml(dataset_path):
    """Corrige el data.yaml para usar rutas relativas."""
    data_yaml_path = Path(dataset_path) / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"❌ No se encontró data.yaml en: {data_yaml_path}")
        return False
    
    print(f"🔧 Corrigiendo data.yaml en: {data_yaml_path}")
    
    # Leer el data.yaml actual
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print("📋 Configuración original:")
    print(f"   path: {data.get('path', 'N/A')}")
    print(f"   train: {data.get('train', 'N/A')}")
    print(f"   val: {data.get('val', 'N/A')}")
    print(f"   test: {data.get('test', 'N/A')}")
    
    # Crear backup
    backup_path = data_yaml_path.parent / "data_original.yaml"
    with open(backup_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"💾 Backup creado: {backup_path}")
    
    # Corregir las rutas para que sean relativas
    data['path'] = '.'  # Directorio actual
    data['train'] = 'train/images'
    data['val'] = 'val/images'
    data['test'] = 'test/images'
    
    # Guardar el data.yaml corregido
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("✅ Configuración corregida:")
    print(f"   path: {data['path']}")
    print(f"   train: {data['train']}")
    print(f"   val: {data['val']}")
    print(f"   test: {data['test']}")
    
    return True

def main():
    # Detectar automáticamente la carpeta del dataset
    dataset_path = Path("Dist/dental_ai/datasets/detection_combined")
    
    if not dataset_path.exists():
        print(f"❌ No se encontró el dataset en: {dataset_path}")
        return
    
    if fix_data_yaml(dataset_path):
        print("🎉 data.yaml corregido exitosamente!")
        print(f"💡 Ahora puedes ejecutar el entrenamiento desde: {dataset_path}")
        print(f"   cd {dataset_path}")
        print("   ./train_dental_dataset.sh")
    else:
        print("❌ Error al corregir data.yaml")

if __name__ == "__main__":
    main()
