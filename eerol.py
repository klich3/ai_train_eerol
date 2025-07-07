#!/usr/bin/env python3
"""
🔧 EEROL - Universal Dataset Management Tool
============================================

Universal tool for computer vision dataset management:
- Scan and analyze datasets from any directory
- Convert between annotation formats (YOLO, COCO, etc.)
- Generate training scripts for different frameworks
- Mix datasets with custom proportions
- Train models with generated scripts

Author: Universal Tool (refactored from dental-specific)
Version: 3.0 (Universal)
"""

import os
import sys
import argparse
from pathlib import Path

# Add local modules to path
sys.path.append(str(Path(__file__).parent))

from eerol.dataset_scanner import DatasetScanner
from eerol.dataset_converter import DatasetConverter
from eerol.dataset_preview import DatasetPreview
from eerol.dataset_splitter import DatasetSplitter
from eerol.script_generator import ScriptGenerator
from eerol.utils import EerolUtils


def print_banner():
    """🎨 Print EEROL banner."""
    banner = """
    ████████ ████████ ██████   ██████  ██      
    ██       ██       ██   ██ ██    ██ ██      
    ██████   █████    ██████  ██    ██ ██      
    ██       ██       ██   ██ ██    ██ ██      
    ████████ ████████ ██   ██  ██████  ████████
    
    🔧 EEROL - Universal Dataset Tool v3.0
    ======================================
    
    ✨ Universal Computer Vision Dataset Manager
    📊 Scan • 🔄 Convert • 🏗️ Generate • 🚀 Train
    """
    print(banner)


def get_base_path():
    """📂 Get base path from user (HOME or current directory)."""
    print("\n📂 SELECCIONAR DIRECTORIO BASE")
    print("=" * 40)
    print("1. Usar directorio HOME ($HOME)")
    print("2. Usar directorio actual")
    print("3. Especificar ruta personalizada")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == "1":
        return Path.home()
    elif choice == "2":
        return Path.cwd()
    elif choice == "3":
        custom_path = input("Introduce la ruta: ").strip()
        path = Path(custom_path).expanduser().resolve()
        if not path.exists():
            print(f"❌ La ruta {path} no existe.")
            return get_base_path()
        return path
    else:
        print("❌ Opción inválida. Inténtalo de nuevo.")
        return get_base_path()


def scan_command(args):
    """🔍 Scan datasets in directory."""
    if args.path:
        base_path = Path(args.path).expanduser().resolve()
    else:
        base_path = get_base_path()
    
    if not base_path.exists():
        print(f"❌ La ruta {base_path} no existe.")
        return
    
    scanner = DatasetScanner(base_path)
    results = scanner.scan_datasets()
    scanner.display_results(results)


def convert_command(args):
    """🔄 Convert dataset format."""
    if not args.input_path or not args.format:
        print("❌ Se requieren --input-path y --format")
        return
    
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"❌ La ruta {input_path} no existe.")
        return
    
    # Create Train directory if not exists
    train_dir = Path.cwd() / "Train" if not args.output else Path(args.output)
    train_dir.mkdir(exist_ok=True)
    
    converter = DatasetConverter(input_path, train_dir)
    result = converter.convert_to_format(args.format, args.name or input_path.name)
    
    if result['success']:
        print(f"✅ Dataset convertido exitosamente a {train_dir / result['output_name']}")
    else:
        print(f"❌ Error en conversión: {result['error']}")


def preview_command(args):
    """👁️ Preview dataset annotations."""
    if not args.image or not args.annotation:
        print("❌ Se requieren --image y --annotation")
        return
    
    image_path = Path(args.image).expanduser().resolve()
    annotation_path = Path(args.annotation).expanduser().resolve()
    
    if not image_path.exists() or not annotation_path.exists():
        print("❌ Archivo de imagen o anotación no existe.")
        return
    
    preview = DatasetPreview()
    preview.show_preview(image_path, annotation_path, args.format or 'yolo')


def split_command(args):
    """✂️ Split dataset with proportions."""
    if not args.input_path:
        print("❌ Se requiere --input-path")
        return
    
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"❌ La ruta {input_path} no existe.")
        return
    
    train_ratio = args.train_ratio or 0.7
    val_ratio = args.val_ratio or 0.3
    test_ratio = args.test_ratio or 0.0
    
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    train_dir = Path.cwd() / "Train" if not args.output else Path(args.output)
    train_dir.mkdir(exist_ok=True)
    
    splitter = DatasetSplitter(input_path, train_dir)
    result = splitter.split_dataset(
        args.name or input_path.name,
        train_ratio, val_ratio, test_ratio
    )
    
    if result['success']:
        print(f"✅ Dataset dividido exitosamente en {result['output_path']}")
        print(f"  📊 Train: {result['train_count']} imágenes")
        print(f"  📊 Val: {result['val_count']} imágenes")
        if result['test_count'] > 0:
            print(f"  📊 Test: {result['test_count']} imágenes")
    else:
        print(f"❌ Error dividiendo dataset: {result['error']}")


def train_command(args):
    """🚀 Train model with generated script."""
    train_dir = Path.cwd() / "Train"
    if not train_dir.exists():
        print("❌ Directorio Train no existe. Crea datasets primero.")
        return
    
    # List available datasets
    datasets = [d for d in train_dir.iterdir() if d.is_dir()]
    if not datasets:
        print("❌ No hay datasets en el directorio Train.")
        return
    
    if args.dataset:
        dataset_name = args.dataset
        dataset_path = train_dir / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset {dataset_name} no encontrado.")
            return
    else:
        print("\n📋 DATASETS DISPONIBLES PARA ENTRENAR:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset.name}")
        
        try:
            choice = int(input("\nSelecciona dataset (número): ")) - 1
            if 0 <= choice < len(datasets):
                dataset_path = datasets[choice]
            else:
                print("❌ Selección inválida.")
                return
        except ValueError:
            print("❌ Entrada inválida.")
            return
    
    # Execute training script
    script_generator = ScriptGenerator(dataset_path)
    result = script_generator.execute_training()
    
    if result['success']:
        print(f"✅ Entrenamiento iniciado para {dataset_path.name}")
    else:
        print(f"❌ Error iniciando entrenamiento: {result['error']}")


def list_train_datasets():
    """📋 List datasets in Train directory."""
    train_dir = Path.cwd() / "Train"
    if not train_dir.exists():
        print("❌ Directorio Train no existe.")
        return
    
    datasets = [d for d in train_dir.iterdir() if d.is_dir()]
    if not datasets:
        print("📁 No hay datasets en el directorio Train.")
        return
    
    print("\n📋 DATASETS EN TRAIN:")
    print("=" * 40)
    for dataset in datasets:
        # Get basic info about dataset
        info = EerolUtils.get_dataset_info(dataset)
        print(f"📁 {dataset.name}")
        print(f"   📊 Imágenes: {info['total_images']}")
        print(f"   🏷️ Formato: {info['format']}")
        print(f"   📝 Script: {'✅' if info['has_script'] else '❌'}")
        print()


def clean_command():
    """🧹 Clean unused files."""
    utils = EerolUtils()
    result = utils.clean_unused_files()
    
    if result['success']:
        print(f"✅ Limpieza completada:")
        print(f"   🗑️ Archivos eliminados: {result['deleted_count']}")
        print(f"   💾 Espacio liberado: {result['space_freed']}")
    else:
        print(f"❌ Error en limpieza: {result['error']}")


def interactive_mode():
    """🎛️ Interactive mode with menu."""
    print_banner()
    
    while True:
        print("\n🎛️ MENÚ PRINCIPAL")
        print("=" * 40)
        print("1. 🔍 Escanear datasets")
        print("2. 🔄 Convertir formato")
        print("3. 👁️ Previsualizar anotaciones")
        print("4. ✂️ Dividir dataset")
        print("5. 📋 Listar datasets Train")
        print("6. 🚀 Entrenar modelo")
        print("7. 🧹 Limpiar archivos")
        print("0. 🚪 Salir")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == "1":
            scan_command(argparse.Namespace(path=None))
        elif choice == "2":
            print("\n🔄 CONVERSIÓN DE FORMATO")
            input_path = input("Ruta del dataset: ").strip()
            format_type = input("Formato destino (yolo/coco): ").strip()
            name = input("Nombre del dataset (opcional): ").strip() or None
            
            args = argparse.Namespace(
                input_path=input_path,
                format=format_type,
                name=name,
                output=None
            )
            convert_command(args)
        elif choice == "3":
            print("\n👁️ PREVISUALIZACIÓN")
            image = input("Ruta de la imagen: ").strip()
            annotation = input("Ruta de la anotación: ").strip()
            format_type = input("Formato (yolo/coco): ").strip() or 'yolo'
            
            args = argparse.Namespace(
                image=image,
                annotation=annotation,
                format=format_type
            )
            preview_command(args)
        elif choice == "4":
            print("\n✂️ DIVISIÓN DE DATASET")
            input_path = input("Ruta del dataset: ").strip()
            train_ratio = float(input("Proporción train (0.7): ").strip() or "0.7")
            val_ratio = float(input("Proporción validation (0.3): ").strip() or "0.3")
            test_ratio = float(input("Proporción test (0.0): ").strip() or "0.0")
            name = input("Nombre del dataset (opcional): ").strip() or None
            
            args = argparse.Namespace(
                input_path=input_path,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                name=name,
                output=None
            )
            split_command(args)
        elif choice == "5":
            list_train_datasets()
        elif choice == "6":
            train_command(argparse.Namespace(dataset=None))
        elif choice == "7":
            clean_command()
        elif choice == "0":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida.")


def main():
    """🚀 Main function."""
    parser = argparse.ArgumentParser(
        description="EEROL - Universal Dataset Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan datasets in directory')
    scan_parser.add_argument('--path', help='Path to scan (optional)')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert dataset format')
    convert_parser.add_argument('--input-path', required=True, help='Input dataset path')
    convert_parser.add_argument('--format', required=True, choices=['yolo', 'coco'], help='Target format')
    convert_parser.add_argument('--name', help='Output dataset name')
    convert_parser.add_argument('--output', help='Output directory (default: ./Train)')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview annotations')
    preview_parser.add_argument('--image', required=True, help='Image file path')
    preview_parser.add_argument('--annotation', required=True, help='Annotation file path')
    preview_parser.add_argument('--format', choices=['yolo', 'coco'], default='yolo', help='Annotation format')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset with proportions')
    split_parser.add_argument('--input-path', required=True, help='Input dataset path')
    split_parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio (default: 0.7)')
    split_parser.add_argument('--val-ratio', type=float, default=0.3, help='Validation ratio (default: 0.3)')
    split_parser.add_argument('--test-ratio', type=float, default=0.0, help='Test ratio (default: 0.0)')
    split_parser.add_argument('--name', help='Output dataset name')
    split_parser.add_argument('--output', help='Output directory (default: ./Train)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--dataset', help='Dataset name to train')
    
    # List command
    subparsers.add_parser('list', help='List datasets in Train directory')
    
    # Clean command
    subparsers.add_parser('clean', help='Clean unused files')
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        scan_command(args)
    elif args.command == 'convert':
        convert_command(args)
    elif args.command == 'preview':
        preview_command(args)
    elif args.command == 'split':
        split_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'list':
        list_train_datasets()
    elif args.command == 'clean':
        clean_command()
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
