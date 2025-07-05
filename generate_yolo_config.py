#!/usr/bin/env python3
"""
🔧 Generador de data.yaml dinámico para YOLO
============================================

Genera un data.yaml con rutas absolutas según el sistema operativo.
"""

import os
import yaml
from pathlib import Path

def generate_yolo_config(dataset_dir: str, output_file: str = None):
    """
    Genera un data.yaml con rutas absolutas para YOLO.
    
    Args:
        dataset_dir: Directorio del dataset detection_combined
        output_file: Archivo de salida (opcional)
    
    Returns:
        str: Ruta del archivo data.yaml generado
    """
    dataset_path = Path(dataset_dir).resolve()
    
    # Verificar que existe el directorio
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    
    # Leer configuración original si existe
    original_config = dataset_path / "data.yaml"
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
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images'
    })
    
    # Verificar que existen las carpetas
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split / 'images'
        if split_path.exists():
            img_count = len(list(split_path.glob('*.jpg')) + 
                          list(split_path.glob('*.jpeg')) + 
                          list(split_path.glob('*.png')))
            print(f"✅ {split}: {img_count} imágenes en {split_path}")
        else:
            print(f"⚠️ {split}: No encontrado {split_path}")
    
    # Archivo de salida
    if output_file is None:
        output_file = "temp_data.yaml"
    
    output_path = Path(output_file).resolve()
    
    # Guardar configuración
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 data.yaml generado: {output_path}")
    print(f"📁 Dataset path: {config['path']}")
    print(f"📊 Clases: {config['nc']}")
    
    return str(output_path)

def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar data.yaml dinámico para YOLO')
    parser.add_argument('--dataset-dir', 
                       default='../datasets/detection_combined',
                       help='Directorio del dataset')
    parser.add_argument('--output', '-o',
                       default='temp_data.yaml',
                       help='Archivo de salida')
    
    args = parser.parse_args()
    
    try:
        # Calcular ruta del dataset relativa al script
        script_dir = Path(__file__).parent
        if args.dataset_dir.startswith('..'):
            dataset_dir = script_dir / args.dataset_dir
        else:
            dataset_dir = Path(args.dataset_dir)
        
        # Generar configuración
        config_file = generate_yolo_config(str(dataset_dir), args.output)
        print(f"✅ Configuración generada exitosamente: {config_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
